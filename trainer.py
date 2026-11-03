import copy
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import Stage1Dataset, Stage2Dataset
from eval import Evaluator
from loss import AppearanceLoss, GeometricLoss
from mcb_metrics import MCBEvaluator
from models.alignment_network import AlignmentNetwork
from preprocessor import Preprocessor
from utilities.panorama import create_panorama_video_direct
from utilities.preprocess_utils import count_videos


def pixel_to_grid(height: int, width: int, device) -> torch.Tensor:
    ''' Maps pixel coordinates [0, width-1] x [0, height-1] (origin at the top-left corner) to the normalized
    coordinates [-1, 1] x [-1, 1] (origin at the image centre) in which the alignment net predicts. '''
    return torch.tensor([
        [2.0 / (width - 1.0), 0.0, -1.0],
        [0.0, 2.0 / (height - 1.0), -1.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)


def get_num_videos(args: Dict[str, Any]) -> int:
    ''' N: 2 on StabStitch-D, from the keypoint annotations on MMCD, the number of video folders on a new scene. '''
    if args['dataset'] == 'StabStitch-D':
        return 2
    if args['dataset'] == 'MMCD':
        scene = args['scene']
        return torch.load(f"{args['data_folder']}/kps_annotations/{scene}/kps_{scene}.pt", map_location='cpu').shape[1]
    return count_videos(Path(args['scene_folder']))


class Trainer:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = args['device']
        self.num_videos = get_num_videos(args)  # N
        self.img_res = args['image_resolution']  # r

        # Sec. 3.1: frames resized to r x r and RGB-normalized, spatial and temporal ORTs
        video_folder = Path(args['scene_folder'])
        self.preprocessor = Preprocessor(video_folder, self.num_videos, self.device,
                                         img_resolution=self.img_res, matcher=args['matcher'],
                                         certainty_threshold=args['certainty_threshold'])
        self.mv_orts, self.temporal_orts, self.frames = self.preprocessor.preprocess()
        # mv_orts [video_len, N, N, 3, 3], temporal_orts [video_len-1, 2, N, 3, 3], frames [N, video_len, 3, r, r]
        
        self.video_len = self.frames.shape[1]

        self.alignment_net = AlignmentNetwork(num_videos=self.num_videos, batch_size=args['batch_size'],
                                              img_size=(self.img_res, self.img_res)).to(self.device)
        print(f'Number of parameters in the alignment net: {sum(p.numel() for p in self.alignment_net.parameters())}, '
              f'trainable: {sum(p.numel() for p in self.alignment_net.parameters() if p.requires_grad)}')

        self.geometric_loss = GeometricLoss(args['stage1_d'], self.device, self.num_videos)
        self.appearance_loss = AppearanceLoss(self.num_videos, device=self.device)

        # PSNR and SSIM on StabStitch-D only; GV, stability and CMSA need the MMCD keypoint annotations
        self.psnr_ssim_evaluator = Evaluator() if args['dataset'] == 'StabStitch-D' else None
        self.mcb_evaluator = MCBEvaluator(self.device, args['data_folder'], args['scene'],
                                          self.preprocessor.org_H, self.preprocessor.org_W) \
            if args['dataset'] == 'MMCD' else None

    def train(self):
        ''' Runs both optimization stages, then evaluates. Returns (psnr, ssim, geo_var, stability, cmsa). '''
        self._optimize(
            'Stage 1 (geometric)',
            Stage1Dataset(self.frames, self.mv_orts, self.temporal_orts, self.video_len),
            self.geometric_loss,
            lr=self.args['stage1_lr'], patience=self.args['stage1_patience'],
            num_epochs=self.args['stage1_epochs'])

        # The overlap masks are only needed from Stage 2 on, so they are computed here to save memory
        spatial_masks, temporal_masks = self.preprocessor.calc_masks()
        windows_with_masks = Stage2Dataset(self.frames, spatial_masks, temporal_masks, self.video_len)
        self._optimize(
            'Stage 2 (appearance)',
            windows_with_masks,
            self.appearance_loss,
            lr=self.args['stage2_lr'], patience=self.args['stage2_patience'],
            num_epochs=self.args['stage2_epochs'])
        torch.cuda.empty_cache()

        self.alignment_net.eval()
        with torch.inference_mode():
            return self.evaluate(windows_with_masks)

    def _optimize(self, stage, dataset, loss_fn, lr, patience, num_epochs):
        '''
        Optimizes the alignment net with Adam and ReduceLROnPlateau on the mean epoch loss,
        then restores the weights of the epoch with the lowest loss.
        dataset: sliding windows, windows['frames'] is [batch_size, 2, N, 3, r, r].
        loss_fn: (warper, windows) -> scalar loss.
        '''
        data_loader = DataLoader(dataset, self.args['batch_size'], shuffle=self.args['shuffle'])
        optimizer = torch.optim.Adam(self.alignment_net.parameters(), lr=lr, weight_decay=self.args['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

        best_loss, best_epoch, best_state = float('inf'), None, None
        self.alignment_net.train()
        progress_bar = tqdm(range(num_epochs), desc=stage, position=0, leave=True)
        for epoch in progress_bar:
            epoch_loss = 0.0
            for windows in data_loader:
                # One homography per frame, frames ordered (window, time, video)
                warper = self.alignment_net(windows['frames'].reshape(-1, 3, self.img_res, self.img_res))
                loss = loss_fn(warper, windows)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.alignment_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(data_loader)

            if epoch_loss < best_loss:
                best_loss, best_epoch, best_state = epoch_loss, epoch, copy.deepcopy(self.alignment_net.state_dict())

            progress_bar.set_description(f'{stage} epoch {epoch + 1}/{num_epochs}', refresh=False)
            progress_bar.set_postfix({'Loss': f'{epoch_loss:.5f}'}, refresh=False)
            scheduler.step(epoch_loss)

        if best_state is not None:
            self.alignment_net.load_state_dict(best_state)
            print(f'\nLoaded the best {stage} weights: epoch {best_epoch + 1}, loss {best_loss:.5f}')

    def evaluate(self, dataset):
        '''
        Sec. 3.6: predicts H_i^t for every frame t that starts a sliding window (all but the last frame), in pixel
        coordinates of the original frames. The solution is defined only up to a global transformation, so it is
        expressed relative to a reference:
            fixed:  (H_0^t)^-1 H_i^t, video 0 stays fixed at every t. Used by all metrics.
            moving: (H_0^0)^-1 H_i^t, the first frame of video 0 is the reference for the whole sequence.
        Returns (psnr, ssim, geo_var, stability, cmsa); metrics that do not apply to the dataset are 0.0.
        '''
        print(f"beginning evaluation for {self.args['scene']}")
        grid_from_pixel = pixel_to_grid(self.preprocessor.org_H, self.preprocessor.org_W, self.device)
        pixel_from_grid = torch.linalg.inv(grid_from_pixel)
        frames_org, _ = self.preprocessor.extract_frames()  # [N, video_len, 3, org_H, org_W]

        fixed_homographies, moving_homographies = [], []
        for windows in DataLoader(dataset, batch_size=1, shuffle=False):
            t = windows['t'].item()
            warper = self.alignment_net(windows['frames'].reshape(-1, 3, self.img_res, self.img_res))
            homographies = warper.get_homographies(inverse=False).reshape(2, self.num_videos, 3, 3)[0]           # H_i^t
            reference_inverse = warper.get_homographies(inverse=True).reshape(2, self.num_videos, 3, 3)[0, 0]    # (H_0^t)^-1
            if t == 0:
                first_reference_inverse = reference_inverse                                                     # (H_0^0)^-1

            fixed = pixel_from_grid @ (reference_inverse @ homographies @ grid_from_pixel)          # [N, 3, 3]
            moving = pixel_from_grid @ (first_reference_inverse @ homographies @ grid_from_pixel)   # [N, 3, 3]
            fixed_homographies.append(fixed)
            moving_homographies.append(moving)

            if self.psnr_ssim_evaluator is not None:
                self.psnr_ssim_evaluator.update_psnr_and_ssim_batch(frames_org[:, t], fixed[1])
            if self.mcb_evaluator is not None:
                self.mcb_evaluator.update_metrics(fixed, t=t)

        self._save_panoramic_videos(frames_org, torch.stack(fixed_homographies, dim=1), torch.stack(moving_homographies, dim=1))

        geo_var, stability, cmsa = self.mcb_evaluator.get_metrics() if self.mcb_evaluator is not None else (0.0, 0.0, 0.0)
        if self.psnr_ssim_evaluator is not None:
            psnr, ssim = self.psnr_ssim_evaluator.get_psnr(), self.psnr_ssim_evaluator.get_ssim()
        else:
            psnr, ssim = 0.0, 0.0
        return psnr, ssim, geo_var, stability, cmsa

    def _save_panoramic_videos(self, frames_org, fixed_homographies, moving_homographies):
        '''
        Saves the video(s) chosen by --panoramic_video_type to <results_folder>/<scene>[_colored].
        frames_org [N, video_len, 3, org_H, org_W], homographies [N, video_len-1, 3, 3].
        '''
        output = self.args['panoramic_video_type']
        if output == 'none':
            return
        scene = self.args['scene']
        colored = self.args['colored_panorama']
        output_dir = Path(self.args['results_folder']) / f"{scene}{'_colored' if colored else ''}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output in ('fixed', 'both'):
            create_panorama_video_direct(frames_org, fixed_homographies, str(output_dir), f'fixed_panorama_{scene}',
                                         fps=self.args['fps'], colored_panorama=colored)
        if output in ('moving', 'both'):
            try:
                create_panorama_video_direct(frames_org, moving_homographies, str(output_dir), f'moving_panorama_{scene}',
                                             fps=self.args['fps'], colored_panorama=colored)
            except Exception as e:
                print(f'Could not create the moving panorama video: {e}')
