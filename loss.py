"""The two losses of MViS (paper Sec. 3.3 and 3.4).

GeometricLoss  = L_geo, the Stage 1 ORT-guided geometric loss (Eq. 1-4).
AppearanceLoss = L_rgb, the Stage 2 appearance loss (Eq. 5-8).

Conventions:
    t_window = 2: every sample is a sliding window of 2 consecutive frames, relative times t=0,1.
    Indices: b = window in the batch, t = relative time in the window (absolute frame = inputs['t'][b] + t),
        i, j = videos, dir = temporal direction (0: fwd, t -> t+1; 1: bwd, t+1 -> t).
    H[b,t,i] is the paper's H_i^t (after the matrix exponential); an ORT is the paper's H-hat.
    All homographies act on normalized image coordinates [-1,1]x[-1,1] (origin at the image centre).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_warper import HomographyWarper


class GeometricLoss(nn.Module):
    """L_geo, the Stage 1 ORT-guided geometric loss (paper Sec. 3.3).

    Args:
        d: the transformations are compared on the d x d grid G.
        device: device of G.
        num_videos: N.
    """

    def __init__(self, d, device, num_videos):
        super().__init__()

        # The smooth-l1 loss of Eq. 1-4 (Huber with delta = 1).
        self.criterion = lambda x, y, dim: F.huber_loss(x, y, reduction='none', delta=1.0).mean(dim=dim)
        self.d = d
        self.device = device
        self.num_videos = num_videos
        self.G = self.make_grid_points_G()
        self.zero3x3 = torch.zeros((3, 3), device=self.device)

    def make_grid_points_G(self):
        """The regular d x d grid G over [-1,1]x[-1,1].

        Returns:
            G: [3, d^2], homogeneous coordinates, rows (x, y, 1).
        """
        x = torch.linspace(-1.0, 1.0, self.d, device=self.device, dtype=torch.float32, requires_grad=False)
        y = torch.linspace(-1.0, 1.0, self.d, device=self.device, dtype=torch.float32, requires_grad=False)

        xv, yv = torch.meshgrid(x, y, indexing='ij')

        G = torch.stack([xv.flatten(), yv.flatten(),
                         torch.ones(self.d * self.d, device=self.device, dtype=torch.float32, requires_grad=False)],
                        dim=1).T

        return G

    def to_inhomogeneous(self, coords_hom):
        return coords_hom[..., :2, :] / coords_hom[..., 2:, :].clamp(min=1e-8)

    def L_geo_time(self, H, H_inv, temporal_orts):
        """L_geo^time = L_geo^fwd + L_geo^bwd (Eq. 2-4).

        Per video, the predicted transformation between consecutive frames should match the temporal ORT.
        Compares temporal_orts[b,t,0,i] with H_inv[b,t+1,i] @ H[b,t,i] (Eq. 3) and temporal_orts[b,t,1,i] with
        H_inv[b,t,i] @ H[b,t+1,i] (Eq. 4), on the grid G.

        Args:
            H: [batch_size, t_window, num_videos, 3, 3]. H[b,t,i] is the homography (after the matrix exponential)
                that warps the frame of window b, at relative time t, of video i to the panoramic domain.
            H_inv: [batch_size, t_window, num_videos, 3, 3]. H_inv[b,t,i] is the inverse of H[b,t,i]: it warps the
                panoramic domain back to the frame of window b, at relative time t, of video i.
            temporal_orts: [batch_size, t_window-1, 2, num_videos, 3, 3]. temporal_orts[b,t,0,i] is the ORT from the
                frame of video i at relative time t to its frame at t+1, in window b. temporal_orts[b,t,1,i] is the
                same for t+1 -> t. A zero matrix means no reliable estimate (it is not masked out here).

        Returns:
            loss_mean (scalar): mean over batch, time, direction, videos and grid points.
        """
        G = self.G  # [3, d^2]

        Htp1_inv_mul_Ht = torch.matmul(H_inv[:, 1:], H[:, :-1])  # [B, T-1, V, 3, 3]
        Ht_inv_mul_Htp1 = torch.matmul(H_inv[:, :-1], H[:, 1:])

        preds = torch.stack([
            Htp1_inv_mul_Ht,
            Ht_inv_mul_Htp1,
        ], dim=2)  # [B, T-1, 2, V, 3, 3]

        preds_n_orts = torch.stack([preds, temporal_orts], dim=0)  # [2, B, T-1, 2, V, 3, 3]
        preds_n_orts_G_hom = torch.matmul(preds_n_orts, G)  # [2, B, T-1, 2, V, 3, d^2]
        # [2, B, T-1, 2, V, d^2, 2]
        preds_n_orts_G = self.to_inhomogeneous(preds_n_orts_G_hom).permute(0, 1, 2, 3, 4, 6, 5)

        temporal_loss = self.criterion(
                preds_n_orts_G[0],  # [B, T-1, 2, V, d^2, 2]
                preds_n_orts_G[1],  # [B, T-1, 2, V, d^2, 2]
                dim=[-4, -2, -1]
            )

        loss_mean = temporal_loss.mean()
        return loss_mean

    def L_geo_views(self, H, H_inv, mv_orts):
        """L_geo^views (Eq. 1).

        For every pair of videos (i, j) at the same time, the predicted transformation i -> j should match the
        multi-view ORT. Compares mv_orts[b,t,i,j] with H_inv[b,t,j] @ H[b,t,i] on the grid G.

        Args:
            H: [batch_size, t_window, num_videos, 3, 3]. H[b,t,i] is the homography (after the matrix exponential)
                that warps the frame of window b, at relative time t, of video i to the panoramic domain.
            H_inv: [batch_size, t_window, num_videos, 3, 3]. H_inv[b,t,j] is the inverse of H[b,t,j]: it warps the
                panoramic domain back to the frame of window b, at relative time t, of video j.
            mv_orts: [batch_size, t_window, num_videos, num_videos, 3, 3]. mv_orts[b,t,i,j] is the ORT from the frame
                of video i to the frame of video j, both at relative time t of window b. Zero matrix on the diagonal
                (i == j) and where no reliable estimate exists; those pairs are skipped.

        Returns:
            loss_mean (scalar): mean over t of the per-time mean over valid (b, i, j) pairs.
        """
        epsilon = 1e-8

        G = self.G  # [3, d^2]

        loss_mean = torch.zeros((), device=self.device, dtype=torch.float32)

        for t in range(2):
            valid_mask_t = torch.any(torch.any(mv_orts[:, t] != self.zero3x3, dim=-1), dim=-1)  # [B, V, V]
            total_loss_t = torch.tensor(0.0, device=self.device)
            count_t = 0
            if valid_mask_t.any():
                b_indices, i_indices, j_indices = torch.where(valid_mask_t)
                num_valid_pairs = i_indices.size(0)

                Hi = H[b_indices, t, i_indices]                             # [num_valid_pairs, 3, 3]
                Hj_inv = H_inv[b_indices, t, j_indices]                     # [num_valid_pairs, 3, 3]
                H_hat_i_to_j = mv_orts[b_indices, t, i_indices, j_indices]  # [num_valid_pairs, 3, 3]

                G_expanded = G.unsqueeze(0).expand(num_valid_pairs, -1, -1)  # [num_valid_pairs, 3, d^2]

                Hj_inv_mul_Hi = torch.bmm(Hj_inv, Hi)  # [num_valid_pairs, 3, 3]

                pred_G_hom = torch.bmm(Hj_inv_mul_Hi, G_expanded)  # [num_valid_pairs, 3, d^2]
                ort_G_hom = torch.bmm(H_hat_i_to_j, G_expanded)    # [num_valid_pairs, 3, d^2]

                denom_pred = pred_G_hom[:, 2:3, :]  # [num_valid_pairs, 1, d^2]
                denom_ort = ort_G_hom[:, 2:3, :]    # [num_valid_pairs, 1, d^2]

                valid_denom_pred = torch.abs(denom_pred) > epsilon
                valid_denom_ort = torch.abs(denom_ort) > epsilon

                pred_G_xy = torch.zeros_like(pred_G_hom[:, :2, :])  # [num_valid_pairs, 2, d^2]
                ort_G_xy = torch.zeros_like(ort_G_hom[:, :2, :])    # [num_valid_pairs, 2, d^2]

                # a point with a near-zero denominator stays (0, 0)
                pred_G_xy = torch.where(
                    valid_denom_pred,
                    pred_G_hom[:, :2, :] / (denom_pred + epsilon * torch.sign(denom_pred)),
                    pred_G_xy
                )

                ort_G_xy = torch.where(
                    valid_denom_ort,
                    ort_G_hom[:, :2, :] / (denom_ort + epsilon * torch.sign(denom_ort)),
                    ort_G_xy
                )

                valid_points = torch.logical_and(valid_denom_pred.squeeze(1), valid_denom_ort.squeeze(1))

                pred_G = pred_G_xy.permute(0, 2, 1)  # [num_valid_pairs, d^2, 2]
                ort_G = ort_G_xy.permute(0, 2, 1)    # [num_valid_pairs, d^2, 2]

                if valid_points.any():

                    batch_loss = self.criterion(pred_G, ort_G, dim=[-2, -1])  # [num_valid_pairs]
                    total_loss_t = batch_loss.sum()
                    count_t = batch_loss.numel()
                if count_t > 0:
                    loss_t = total_loss_t / count_t
                else:
                    loss_t = torch.tensor(0.0, device=self.device)

                loss_mean = (t * loss_mean + loss_t) / (t + 1)

        return loss_mean

    def forward(self, warper: HomographyWarper, inputs):
        """L_geo = L_geo^views + 50 * L_geo^time.

        Args:
            warper: the HomographyWarper returned by the alignment net for frames.reshape(-1, 3, res, res), i.e.
                batch_size*t_window*num_videos frames ordered (b, t, i). Its homographies are reshaped here to
                [batch_size, t_window, num_videos, 3, 3].
            inputs: a Stage1Dataset batch.
                inputs['mv_orts']: [batch_size, t_window, num_videos, num_videos, 3, 3], see L_geo_views.
                inputs['temporal_orts']: [batch_size, t_window-1, 2, num_videos, 3, 3], see L_geo_time.

        Returns:
            L_geo (scalar).
        """
        mv_orts = inputs['mv_orts']
        temporal_orts = inputs['temporal_orts']
        H = warper.get_homographies(inverse=False).reshape(-1, 2, self.num_videos, 3, 3)
        H_inv = warper.get_homographies(inverse=True).reshape(-1, 2, self.num_videos, 3, 3)
        H_clone = H.clone()
        H_inv_clone = H_inv.clone()
        L_geo_time = self.L_geo_time(H_clone, H_inv_clone, temporal_orts)
        L_geo_views = self.L_geo_views(H, H_inv, mv_orts)
        L_geo = L_geo_views + 50 * L_geo_time
        return L_geo


class AppearanceLoss(nn.Module):
    """L_rgb, the Stage 2 appearance loss (paper Sec. 3.4).

    Args:
        num_videos: N.
        device: device of the returned loss.
    """

    def __init__(self, num_videos, device):
        super().__init__()

        # The squared l2 error of Eq. 5-8.
        self.criterion = lambda x, y, dim: F.mse_loss(x, y, reduction='none').mean(dim=dim)
        self.device = device
        self.num_videos = num_videos

    def L_rgb(self, warped_mv_frames, warped_temporal_frames, frames, mv_masks, temporal_masks):
        """L_rgb = L_rgb^views + L_rgb^time (Eq. 5-8).

        Masked error between the warped frames and the frames they are warped onto, across videos (multi-view) and
        across consecutive times (temporal). Compares warped_mv_frames[b,t,i,j] with frames[b,t,j] (Eq. 5),
        warped_temporal_frames[b,t,0,i] with frames[b,t+1,i] (Eq. 7) and warped_temporal_frames[b,t,1,i] with
        frames[b,t,i] (Eq. 8).

        Args:
            warped_mv_frames: [batch_size, t_window, num_videos, num_videos, 3, res, res]. warped_mv_frames[b,t,i,j] is
                the frame of window b, at relative time t, of video i, warped into the view of video j at the same
                time (sampled with H_inv[b,t,i] @ H[b,t,j]).
                Why not H_inv[b,t,j] @ H[b,t,i], the i -> j mapping used in L_geo_views? grid_sample warps backwards:
                for every output pixel p (in the frame of video j) it needs the location in the source frame
                (video i) to read from. So the grid maps j -> i: H[b,t,j] takes p from frame j to the panoramic
                domain, and H_inv[b,t,i] takes it from there to frame i. This is the inverse of the i -> j mapping:
                (H_inv[b,t,j] @ H[b,t,i])^-1 = H_inv[b,t,i] @ H[b,t,j]. The same holds for the temporal grids below.
            warped_temporal_frames: [batch_size, t_window-1, 2, num_videos, 3, res, res].
                warped_temporal_frames[b,t,0,i] is the frame of video i at relative time t warped into its view at
                t+1 (sampled with H_inv[b,t,i] @ H[b,t+1,i]). warped_temporal_frames[b,t,1,i] is the frame at t+1
                warped into the view at t.
            frames: [batch_size, t_window, num_videos, 3, res, res]. frames[b,t,i] is the paper's I_i^t: the frame of
                window b, at relative time t, of video i, RGB resized to res x res and standardized per channel with
                the mean/std over all frames of all videos.
            mv_masks: [batch_size, t_window, num_videos, num_videos, res, res]. mv_masks[b,t,i,j] is the region of the
                frame of video j covered by the frame of video i (an all-ones image warped with the ORT i -> j,
                values 0/255). Zero for i == j and where no ORT exists.
            temporal_masks: [batch_size, t_window-1, 2, num_videos, res, res]. temporal_masks[b,t,0,i] is the region of
                the frame of video i at t+1 covered by its frame at t. temporal_masks[b,t,1,i] is the region of the
                frame at t covered by the frame at t+1.

        Returns:
            L_rgb (scalar): masked mean of the multi-view error + masked mean of the temporal error.
        """
        mv_target_frames = frames.unsqueeze(2).repeat(1, 1, self.num_videos, 1, 1, 1, 1)  # [B, T, V, V, 3, res, res]
        mv_loss = self.criterion(warped_mv_frames, mv_target_frames, dim=[-3])
        mv_masked_loss = mv_loss * mv_masks

        mv_valid_mask_sum = mv_masks.sum()
        if mv_valid_mask_sum > 0:
            L_rgb = mv_masked_loss.sum() / mv_valid_mask_sum
        else:
            L_rgb = torch.tensor(0.0, device=self.device)

        fwd_target_frames = frames[:, 1:].clone()   # [B, T-1, V, 3, res, res]
        bwd_target_frames = frames[:, :-1].clone()  # [B, T-1, V, 3, res, res]
        # [B, T-1, 2, V, 3, res, res]
        temporal_target_frames = torch.stack([fwd_target_frames, bwd_target_frames], dim=2)
        temporal_loss = self.criterion(warped_temporal_frames, temporal_target_frames, dim=[-3])
        temporal_masked_loss = temporal_loss * temporal_masks
        temporal_valid_mask_sum = temporal_masks.sum()
        if temporal_valid_mask_sum > 0:
            L_rgb += temporal_masked_loss.sum() / temporal_valid_mask_sum

        return L_rgb

    def forward(self, warper: HomographyWarper, inputs):
        """L_rgb of the frames warped by the predicted homographies.

        Args:
            warper: the HomographyWarper returned by the alignment net for frames.reshape(-1, 3, res, res), i.e.
                batch_size*t_window*num_videos frames ordered (b, t, i).
            inputs: a Stage2Dataset batch.
                inputs['frames']: [batch_size, t_window, num_videos, 3, res, res], see L_rgb.
                inputs['mv_masks']: [batch_size, t_window, num_videos, num_videos, res, res], see L_rgb.
                inputs['temporal_masks']: [batch_size, t_window-1, 2, num_videos, res, res], see L_rgb.

        Returns:
            L_rgb (scalar).
        """
        frames = inputs['frames']
        mv_masks = inputs['mv_masks']
        temporal_masks = inputs['temporal_masks']
        warped_mv_frames, warped_temporal_frames = warper(frames)

        return self.L_rgb(warped_mv_frames, warped_temporal_frames, frames, mv_masks, temporal_masks)
