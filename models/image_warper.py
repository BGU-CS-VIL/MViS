from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageWarper(nn.Module):
    """Base class for grid creation and image sampling."""

    def __init__(self, img_size, num_videos):
        super().__init__()
        self.img_size = img_size  # (H, W)
        self.num_videos = num_videos

    def _make_initial_grid(self, shape: List[int], device) -> torch.Tensor:
        """Uniform grid over [-1, 1]x[-1, 1], (x, y) per pixel.

        Args:
            shape (List[int]): (N, _, H, W)

        Returns:
            torch.Tensor: (N, H, W, 2)
        """
        N, _, H, W = shape
        y, x = torch.meshgrid([torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device)])
        grid = torch.stack([x, y], dim=-1)           # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H, W, 2)
        return grid

    def warp_image(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected x to have 4 dimensions, but got {x.dim()}"
        assert x.shape[0] == grid.shape[0], \
            f"Expected grid to have same batch size as x, but got {grid.shape[0]} and {x.shape[0]}"
        assert grid.shape[-3:-1] == x.shape[-2:], \
            f"Expected grid to have shape (N, {x.shape[-2]}, {x.shape[-1]}, 2), but got {grid.shape}"

        return F.grid_sample(x, grid, align_corners=False)


class HomographyWarper(ImageWarper):
    """
    Performs homography-based spatial and temporal warping.

    thetas: [batch_size*t_window*num_videos, 3, 3], zero-trace matrices (before the matrix exponential),
            frames ordered (window, time, video).
    self.homographies = exp(thetas), self.inv_homographies = exp(-thetas) = self.homographies^-1.
    """

    def __init__(self, img_size, thetas: torch.Tensor, num_videos, batch_size):
        super().__init__(img_size, num_videos)

        self.homographies = torch.matrix_exp(thetas)
        self.inv_homographies = torch.matrix_exp(-thetas)

        self.img_size = img_size
        self.device = self.homographies.device
        H, W = self.img_size
        self.batch_size = batch_size
        self.num_frames = self.batch_size * 2 * self.num_videos

        initial_grid = self._make_initial_grid([self.num_frames, 1, H, W], device=self.device)
        initial_grid = initial_grid.permute(0, 3, 1, 2).reshape(self.num_frames, 2, -1)
        # homogeneous coordinates: [num_frames, 3, H*W]
        self.initial_grid = torch.cat([
            initial_grid,
            torch.ones(self.num_frames, 1, initial_grid.shape[-1], device=initial_grid.device),
        ], dim=1)

    def get_homographies(self, inverse):
        if inverse:
            return self.inv_homographies
        else:
            return self.homographies

    def make_temporal_grids(self) -> torch.Tensor:
        """
        Returns:
            temporal_grids: [batch_size*(t_window-1)*2*num_videos, H, W, 2], ordered (b, t, d, i).
                d=0 is the grid warping video i at time t -> time t+1.
                d=1 is the grid warping video i at time t+1 -> time t.
        """
        homographies = self.get_homographies(inverse=False)     # [B*T*V, 3, 3]
        inv_homographies = self.get_homographies(inverse=True)  # [B*T*V, 3, 3]
        num_frames = homographies.shape[0]
        H, W = self.img_size

        inv_homographies = inv_homographies.reshape(-1, 2, self.num_videos, 3, 3)
        batch_size = inv_homographies.shape[0]

        # the last batch of an epoch can be smaller than self.batch_size
        if self.num_frames == num_frames:
            initial_grid = self.initial_grid
        else:
            initial_grid = self._make_initial_grid([num_frames, 1, H, W], device=self.device)
            initial_grid = initial_grid.permute(0, 3, 1, 2).reshape(num_frames, 2, -1)
            initial_grid = torch.cat([
                initial_grid,
                torch.ones(num_frames, 1, initial_grid.shape[-1], device=initial_grid.device),
            ], dim=1)

        grid_t = torch.bmm(homographies, initial_grid)  # [B*T*V, 3, H*W]
        grid_t = grid_t.reshape(batch_size, 2, self.num_videos, 3, H * W)

        temporal_grids = torch.zeros((batch_size, 1, 2, self.num_videos, H, W, 2), device=self.device)

        # t -> t+1
        fwd = torch.matmul(inv_homographies[:, :-1], grid_t[:, 1:])  # [B, T-1, V, 3, H*W]
        fwd_xy = fwd[..., :2, :] / (fwd[..., 2:3, :] + 1e-8)
        fwd_xy = fwd_xy.reshape(batch_size, 1, self.num_videos, 2, H, W)
        fwd_xy = fwd_xy.permute(0, 1, 2, 4, 5, 3)
        temporal_grids[:, :, 0] = fwd_xy

        # t+1 -> t
        bwd = torch.matmul(inv_homographies[:, 1:], grid_t[:, :-1])
        bwd_xy = bwd[..., :2, :] / (bwd[..., 2:3, :] + 1e-8)
        bwd_xy = bwd_xy.reshape(batch_size, 1, self.num_videos, 2, H, W)
        bwd_xy = bwd_xy.permute(0, 1, 2, 4, 5, 3)
        temporal_grids[:, :, 1] = bwd_xy

        temporal_grids = temporal_grids.reshape(-1, H, W, 2)

        return temporal_grids

    def make_mv_grids(self) -> torch.Tensor:
        """
        Returns:
            mv_grids: [batch_size*t_window*num_videos*num_videos, H, W, 2], ordered (b, t, i, j).
                The grid warping video i into the view of video j, both at time t.
        """
        homographies = self.get_homographies(inverse=False)     # [B*T*V, 3, 3]
        inv_homographies = self.get_homographies(inverse=True)  # [B*T*V, 3, 3]

        num_frames, H, W = homographies.shape[0], self.img_size[0], self.img_size[1]

        batch_size = num_frames // (self.num_videos * 2)

        # the last batch of an epoch can be smaller than self.batch_size
        if self.num_frames == num_frames:
            initial_grid = self.initial_grid
        else:
            initial_grid = self._make_initial_grid([num_frames, 1, H, W], device=self.device)
            initial_grid = initial_grid.permute(0, 3, 1, 2).reshape(num_frames, 2, -1)
            initial_grid = torch.cat([
                initial_grid,
                torch.ones(num_frames, 1, initial_grid.shape[-1], device=initial_grid.device),
            ], dim=1)

        grid_t = torch.bmm(homographies, initial_grid)  # [B*T*V, 3, H*W]

        grid_t = grid_t.reshape(batch_size, 2, self.num_videos, 3, -1)                     # [B, T, V, 3, H*W]
        inv_homographies = inv_homographies.reshape(batch_size, 2, self.num_videos, 3, 3)  # [B, T, V, 3, 3]

        # inv_homographies[b,t,i] is applied to the grid of every video j
        inv_homographies_expanded = inv_homographies.unsqueeze(3)  # [B, T, V, 1, 3, 3]
        inv_homographies_expanded = inv_homographies_expanded.expand(
            -1, -1, -1, self.num_videos, -1, -1)  # [B, T, V, V, 3, 3]

        grid_t_expanded = grid_t.unsqueeze(2)  # [B, T, 1, V, 3, H*W]
        grid_t_expanded = grid_t_expanded.expand(-1, -1, self.num_videos, -1, -1, -1)  # [B, T, V, V, 3, H*W]

        hom_grids = torch.matmul(inv_homographies_expanded, grid_t_expanded)  # [B, T, V, V, 3, H*W]

        hom_grids = hom_grids.reshape(-1, 3, H * W)  # [B*T*V*V, 3, H*W]

        xy = hom_grids[:, :2, :]
        z = hom_grids[:, 2:3, :] + 1e-8
        transformed = xy / z  # [B*T*V*V, 2, H*W]

        new_grid = transformed.permute(0, 2, 1).reshape(batch_size * 2 * self.num_videos * self.num_videos, H, W, 2)

        return new_grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): frames, [batch_size, t_window, num_videos, 3, img_res, img_res]

        Returns:
            warped_mv_frames:       [batch_size, t_window, num_videos, num_videos, 3, h, w]
            warped_temporal_frames: [batch_size, t_window-1, 2, num_videos, 3, h, w]
        """
        assert x.dim() == 6, \
            f"Expected x to have 6 dimensions, but got {x.dim()} dimensions " \
            f"(shape: {x.shape}!= [batch_size, t_window, num_videos, 3, img_res, img_res])"
        assert x.shape[2] == self.num_videos, f"Expected x.shape[2] == num_videos, but got x.shape {x.shape}"
        mv_grids = self.make_mv_grids()  # [batch_size*t_window*num_videos*num_videos, H, W, 2]

        res = self.img_size[0]
        assert mv_grids.dim() == 4, \
            f"Expected grid to have 4 dimensions, but got {mv_grids.dim()} dimensions (shape: {mv_grids.shape})"
        # [batch_size*t_window*num_videos*num_videos, 3, res, res]
        mv_source_frames = x.clone().unsqueeze(3).repeat(1, 1, 1, self.num_videos, 1, 1, 1).reshape(-1, 3, res, res)

        warped_mv_frames = self.warp_image(mv_source_frames, mv_grids).reshape(
            -1, 2, self.num_videos, self.num_videos, 3, res, res)

        temporal_grids = self.make_temporal_grids()  # [batch_size*(t_window-1)*2*num_videos, H, W, 2]

        fwd_source_frames = x[:, :-1]  # frames at t
        bwd_source_frames = x[:, 1:]   # frames at t+1
        # [batch_size*(t_window-1)*2*num_videos, 3, h, w]
        temporal_source_frames = torch.cat([fwd_source_frames, bwd_source_frames], dim=2).reshape(-1, 3, res, res)

        warped_temporal_frames = self.warp_image(temporal_source_frames, temporal_grids).reshape(
            -1, 1, 2, self.num_videos, 3, res, res)
        return warped_mv_frames, warped_temporal_frames
