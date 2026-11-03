import torch
import numpy as np
import cv2

from utilities.preprocess_utils import extract_frames
from orts import RoMaMatcher, LoFTRMatcher, OrtEstimator

MATCHERS = {'RoMa': RoMaMatcher, 'LoFTR': LoFTRMatcher}

RNG_DRAWS = 42_884_736


def advance_cpu_rng(n):
    while n > 0:
        k = min(n, 1 << 22)
        torch.empty(k).uniform_()
        n -= k


class Preprocessor:
    def __init__(self, data_folder, num_videos,
                 device, img_resolution, matcher='RoMa', certainty_threshold=0.3):
        self.device = device
        self.data_folder = data_folder
        self.num_videos = num_videos
        self.video_len = -1
        self.img_res = img_resolution
        advance_cpu_rng(RNG_DRAWS)

        self.ort_estimator = OrtEstimator(MATCHERS[matcher](self.device), self.img_res,
                                          certainty_threshold=certainty_threshold)
        self.org_H = None
        self.org_W = None
        
        self.all_spatial_square = None
        self.all_temporal_square = None

    
    def _calc_mask_from_1to2(self, H12_square):
        ''''H12_square: homography matrix from img1 to img2 in torch, img_res: resolution of the image'''
        if not H12_square.any():  # no ORT: cv2.warpPerspective would return a full mask for a zero matrix
            return torch.zeros((self.img_res, self.img_res), dtype=torch.uint8)
        H12_square_np = H12_square.cpu().numpy()
        mask1 = np.ones((self.img_res, self.img_res), dtype=np.uint8) * 255
        warp_mask1 = cv2.warpPerspective(mask1, H12_square_np, (self.img_res, self.img_res), flags=cv2.INTER_NEAREST)
        mask = torch.from_numpy(warp_mask1)
        return mask
        
    def _calc_spatial_masks(self, mv_orts_square):
        ''' mv_orts_square: [num_videos, num_videos, 3, 3]
        return masks: [num_videos, num_videos, img_res, img_res]'''
        masks = torch.zeros((self.num_videos, self.num_videos, self.img_res, self.img_res), device=self.device)
        for i in range(self.num_videos):
            for j in range(self.num_videos):
                if i != j:
                    H12_square = mv_orts_square[i,j]
                    mask = self._calc_mask_from_1to2(H12_square)
                    masks[i,j] = mask
        return masks
    
    def _calc_temporal_masks(self, temporal_orts_square):
        ''' temporal_orts_square: [2, num_videos, 3, 3]
        return masks: [2, num_videos, img_res, img_res]'''
        masks = torch.zeros((2, self.num_videos, self.img_res, self.img_res), device=self.device)
        for i in range(2):
            for j in range(self.num_videos):
                H12_square = temporal_orts_square[i,j]
                mask = self._calc_mask_from_1to2(H12_square)
                masks[i,j] = mask
        return masks
    
    def calc_masks(self):
        all_spatial_masks = torch.zeros((self.video_len, self.num_videos, self.num_videos, self.img_res, self.img_res), device=self.device)
        all_temporal_masks = torch.zeros((self.video_len-1, 2, self.num_videos, self.img_res, self.img_res), device=self.device)
        
        for t in range(self.video_len): 
            all_spatial_masks[t] = self._calc_spatial_masks(self.all_spatial_square[t])
        
        for t in range(self.video_len-1):
            all_temporal_masks[t] = self._calc_temporal_masks(self.all_temporal_square[t])
        return all_spatial_masks, all_temporal_masks
        
    
    
    def _calc_orts(self, frames_paths):
        ''' 
        calulate spatial and temporal orts using the matcher + RANSAC pipline
        inputs:
        frames_paths numpy arr of strings paths first dim is num_videos, second dim is time
        
        returns 
        all_mv_orts  [video_len, num_videos, num_videos, 3, 3]
        all_temporal_orts [video_len-1, 2, num_videos, 3, 3]
        '''
        all_mv_orts  = torch.zeros((self.video_len, self.num_videos, self.num_videos, 3, 3), device=self.device)   
        all_temporal_orts = torch.zeros((self.video_len-1, 2, self.num_videos, 3, 3), device=self.device)    
        all_spatial_square = torch.zeros((self.video_len, self.num_videos, self.num_videos, 3, 3), device=self.device)
        all_temporal_square = torch.zeros((self.video_len-1, 2, self.num_videos, 3, 3), device=self.device)
        
        for t in range(self.video_len):
            all_mv_orts[t], all_spatial_square[t] = self.ort_estimator.create_mv_orts(frames_paths[:,t], t)
                
        for t in range(self.video_len-1):
            all_temporal_orts[t], all_temporal_square[t] = self.ort_estimator.create_temporal_orts(frames_paths[:,t:t+2], t) 
        
        self.all_spatial_square = all_spatial_square
        self.all_temporal_square = all_temporal_square
                
            
        return all_mv_orts, all_temporal_orts
    
    
    def extract_frames(self):
        frames_orgs, frames_square, _ = extract_frames(self.data_folder, self.num_videos, self.img_res, self.device)
        return frames_orgs, frames_square 
       
        
    def preprocess(self):
        
        frames_orgs, frames_square, files = extract_frames(self.data_folder, self.num_videos, self.img_res, self.device)
        # frames_orgs shape [num_videos, video_len, 3, h, w], frames_square [num_videos, video_len, 3, img_res, img_res]
        self.org_H, self.org_W = frames_orgs.shape[-2], frames_orgs.shape[-1]
        self.video_len = frames_square.shape[1]
    
        all_mv_orts, all_temporal_orts = self._calc_orts(files)
        x = frames_square
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        # --- compute dataset stats over [videos, time, H, W]
        data_mean = x.mean(dim=(0, 1, 3, 4))                      # [3]
        data_std  = x.std (dim=(0, 1, 3, 4), unbiased=False)      # [3]

        # --- normalize 
        eps = 1e-6
        data_std_safe = data_std.clamp_min(eps)
        norm_frames_square = (x - data_mean.view(1, 1, 3, 1, 1)) / data_std_safe.view(1, 1, 3, 1, 1)

            
        return all_mv_orts, all_temporal_orts, norm_frames_square
    