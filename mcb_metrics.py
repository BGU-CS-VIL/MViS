"""The Multi-Camera Benchmark (MCB) metrics (paper Sec. 5.1 and Appendix E).

GV:             Geometric Variance, S_GV (Appendix E.1, Eq. 10-14). Lower is better.
StabilityScore: the stability score of Nie et al., "Dynamic Video Stitching via Shakiness Removing", TIP 2018
                (their Eq. 11), on the keypoint trajectories. In (0, 1], higher is better.
CMSA:           Mean Square Acceleration of the video-corner trajectories (Sec. 5.1). Lower is better.

Conventions:
    T: [num_videos, 3, 3]. T[i] is the paper's T^{t,i}: it warps homogeneous pixel coordinates (x, y, 1) of the
        original H x W frame of video i at time t (origin at the top-left corner, x in [0, W-1], y in [0, H-1]) to
        the panoramic space. For MViS, T[i] = (H_0^t)^-1 H_i^t in pixel coordinates (trainer.py), so T[0] is the
        identity.
    p: the keypoints p_m^{t,i}, in pixels of the original frames. b: their visibility flags b_m^{t,i} (1 = visible).
    The frames are scored in order, one update_metrics call per frame t = 0, 1, ...; the trainer scores the frames
    0 .. video_len-2 (one per sliding window).
"""
import math

import torch


class MCBEvaluator:
    """Warps the keypoints once per frame and feeds them to each metric.

    Every metric implements update(T_mul_p, b_t, ...), compute() -> float and reset().
    CMSA is the exception: its update takes T, since it warps the frame corners and not the keypoints.

    Args:
        device: device of the annotations and of the metric states.
        mmcd_folder: MMCD folder.
        scene_name: MMCD scene; its annotations are read from <mmcd_folder>/kps_annotations/<scene_name>/.
        H: height of the original frames, in pixels.
        W: width of the original frames, in pixels.
    """

    def __init__(
        self,
        device: torch.device | str,
        mmcd_folder: str,
        scene_name: str,
        H: int,
        W: int,
    ) -> None:
        # [video_len, num_videos, n_p, 2]
        p = torch.load(f'{mmcd_folder}/kps_annotations/{scene_name}/kps_{scene_name}.pt').to(device)
        # Concatenate a column of ones to make the coordinates homogeneous
        self.p = torch.cat((p, torch.ones_like(p[..., :1])), dim=-1)  # [video_len, num_videos, n_p, 3]
        # Visibility flags from CoTracker3: 1 if the keypoint is visible in that video's frame, 0 otherwise.
        # [video_len, num_videos, n_p]
        self.b = torch.load(f'{mmcd_folder}/kps_annotations/{scene_name}/pred_visibility_{scene_name}.pt').to(device)

        _, num_videos, n_p, _ = p.shape
        self.gv = GV(H, W)
        self.stability = StabilityScore(device, num_videos, n_p)
        self.cmsa = CMSA(H, W)

    def _warp_kps(
        self,
        T: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """T^{t,i} * p_m^{t,i} for every video i and keypoint m.

        Args:
            T: (N, 3, 3), one transformation per video.
            t: frame index.

        Returns:
            (N, n_p, 2): the keypoints of frame t warped by T, in pixels.
        """
        T_mul_p = torch.bmm(T, self.p[t].transpose(1, 2))  # (N, 3, n_p)
        T_mul_p = T_mul_p[:, :2, :] / T_mul_p[:, 2:3, :]   # (N, 2, n_p)
        return T_mul_p.permute(0, 2, 1)                    # (N, n_p, 2)

    @torch.no_grad()
    def update_metrics(
        self,
        T: torch.Tensor,
        t: int,
    ) -> None:
        """Adds frame t to the three metrics.

        Args:
            T: (N, 3, 3), one transformation per video.
            t: frame index.
        """
        T_mul_p = self._warp_kps(T, t)
        b_t = self.b[t]  # (N, n_p)
        self.gv.update(T_mul_p, b_t)
        self.stability.update(T_mul_p, b_t, t)
        self.cmsa.update(T)

    def get_metrics(self) -> tuple[float, float, float]:
        """Returns (gv, stability, cmsa)."""
        return self.gv.compute(), self.stability.compute(), self.cmsa.compute()



class GV:
    """Geometric Variance, S_GV (paper Appendix E.1, Eq. 10-14).

    Args:
        H: height of the original frames, in pixels.
        W: width of the original frames, in pixels.
    """

    def __init__(
        self,
        H: int,
        W: int,
    ) -> None:
        self.diag = math.sqrt(H**2 + W**2)
        self.reset()

    def reset(self) -> None:
        self.sum_mean_variance = 0.0
        self.v_hat_l = 0

    def update(
        self,
        T_mul_p: torch.Tensor,
        b_t: torch.Tensor,
    ) -> None:
        """Adds frame t. A frame without a valid keypoint is skipped (Eq. 12).

        Args:
            T_mul_p: (N, n_p, 2), the keypoints of frame t after warping, in pixels.
            b_t: (N, n_p), visibility in frame t.
        """
        # A keypoint is valid only if it is visible in all videos (Eq. 10)
        M_mt = b_t.all(dim=0)  # (n_p,) bool
        if M_mt.any():
            # Variance across videos, population version (divide by N) (Eq. 13)
            var_mt = T_mul_p.var(dim=0, unbiased=False)  # (n_p, 2)
            var_bar_mt = var_mt.sum(dim=1)               # (n_p,)
            self.sum_mean_variance += var_bar_mt[M_mt].mean().item()
            self.v_hat_l += 1

    def compute(self) -> float:
        """Returns S_GV (Eq. 14): 1000 / diag * sqrt(mean variance), dimensionless."""
        mean_variance = self.sum_mean_variance / max(self.v_hat_l, 1)
        return 1000 * math.sqrt(mean_variance) / self.diag



class StabilityScore:
    """The stability score of Nie et al., "Dynamic Video Stitching via Shakiness Removing", TIP 2018 (their Eq. 11).

    A trajectory is the points p^t = (x, y, t) of one warped keypoint. Over a segment of segment_len frames its
    score is ||p^te - p^ts|| / sum_t ||p^t - p^(t-1)||: the straight line from the first to the last point, divided
    by the length of the curve (1 = stablest).
    Our adjustment for the visibility flags of CoTracker3: the segments are the fixed 40-frame windows 0-39,
    40-79, ..., and within each window a keypoint is used only if it is visible in every frame.

    Args:
        device: device of the state tensors.
        num_videos: N.
        n_p: number of keypoints.
        segment_len: 40, as defined by Nie et al.
    """

    def __init__(
        self,
        device: torch.device | str,
        num_videos: int,
        n_p: int,
        segment_len: int = 40,
    ) -> None:
        self.num_videos = num_videos
        self.n_p = n_p
        self.segment_len = segment_len
        self.device = device
        self.reset()

    def reset(self) -> None:
        # Attributes across segments
        # num_of_valid_segments[i,m] is the number of valid segments of keypoint m in video i
        self.num_of_valid_segments = torch.zeros(self.num_videos, self.n_p, device=self.device)  # [num_videos, n_p]
        # cumulative_stability[i,m] is the sum of the scores over all valid segments of keypoint m in video i
        self.cumulative_stability = torch.zeros(self.num_videos, self.n_p, device=self.device)  # [num_videos, n_p]
        self._reset_segment()

    def _reset_segment(self) -> None:
        # Attributes for the current segment. Notation: pt is the p^t of Nie et al.
        self.visible_throughout_segment = torch.ones(self.num_videos, self.n_p, device=self.device)  # [num_videos, n_p]
        # pt_in_segment has (warped_x, warped_y, t): [num_videos, n_p, segment_len, 3]
        self.pt_in_segment = torch.zeros(self.num_videos, self.n_p, self.segment_len, 3, device=self.device)

    def update(
        self,
        T_mul_p: torch.Tensor,
        b_t: torch.Tensor,
        t: int,
    ) -> None:
        """Adds frame t; scores the segment when t is its last frame.

        Args:
            T_mul_p: [num_videos, n_p, 2], the keypoints of frame t after warping, in pixels.
            b_t: [num_videos, n_p], visibility in frame t.
            t: frame index.
        """
        idx = t % self.segment_len  # index in the segment

        self.pt_in_segment[:, :, idx, :2] = T_mul_p
        self.pt_in_segment[:, :, idx, 2] = t
        # Our adjustment for the visibility flags
        self.visible_throughout_segment = self.visible_throughout_segment * b_t  # [num_videos, n_p]

        # Score immediately after storing the segment's last frame.
        if idx == self.segment_len - 1:
            self._score_segment()
            self._reset_segment()

    def _score_segment(self) -> None:
        seg = self.pt_in_segment  # [num_videos, n_p, segment_len, 3]
        direct_distance = torch.norm(seg[:, :, -1] - seg[:, :, 0], p=2, dim=-1)  # [num_videos, n_p]
        steps = torch.norm(seg[:, :, 1:] - seg[:, :, :-1], p=2, dim=-1)          # [num_videos, n_p, segment_len-1]
        trajectory_length = steps.sum(dim=-1)                                    # [num_videos, n_p]

        # Avoid division by zero
        stability_score = torch.where(trajectory_length < 1e-8,
                                      (direct_distance < 1e-8).float(),
                                      direct_distance / trajectory_length.clamp_min(1e-8))
        valid = self.visible_throughout_segment == 1
        self.cumulative_stability += stability_score * valid
        self.num_of_valid_segments += valid

    def compute(self) -> float:
        """Per video: the sum of the scores of all keypoints / the number of valid segments. Then the mean over the
        N videos.

        Returns:
            The stability score, in (0, 1].
        """
        sum_stability = self.cumulative_stability.sum(dim=1)        # [num_videos]
        num_valid_segments = self.num_of_valid_segments.sum(dim=1)  # [num_videos]
        per_video = torch.where(num_valid_segments > 0, sum_stability / num_valid_segments.clamp_min(1), 0.0)
        return per_video.sum().item() / self.num_videos



class CMSA:
    """Mean Square Acceleration of the video-corner trajectories (CMSA), paper Sec. 5.1.

    The 4 corners of the original frame of each video are warped by T at every frame t;
    acceleration a_t = c_{t+1} - 2 c_t + c_{t-1};
    CMSA = mean over the videos 1..N-1 of the mean of ||a_t||^2 over corners and frames.

    Args:
        H: height of the original frames, in pixels.
        W: width of the original frames, in pixels.
    """

    def __init__(
        self,
        H: int,
        W: int,
    ) -> None:
        # (x, y, 1) of the corners of a W x H frame
        self.corners = torch.tensor([[0, 0, 1], [0, H - 1, 1], [W - 1, 0, 1], [W - 1, H - 1, 1]],
                                    dtype=torch.float32)  # (4, 3)
        self.reset()

    def reset(self) -> None:
        self.corner_trajectory = []  # one (N, 4, 2) tensor per frame

    def update(
        self,
        T: torch.Tensor,
    ) -> None:
        """Adds a frame.

        Args:
            T: (N, 3, 3), one transformation per video.
        """
        warped = torch.matmul(T, self.corners.to(T).T)          # (N, 3, 4)
        warped = warped[:, :2, :] / warped[:, 2:3, :]           # (N, 2, 4)
        self.corner_trajectory.append(warped.permute(0, 2, 1))  # (N, 4, 2)

    def compute(self) -> float:
        """Returns CMSA, in squared pixels of the original frames (0.0 with fewer than 3 frames)."""
        if len(self.corner_trajectory) < 3:
            return 0.0
        # Video 0 is excluded: the fixed homographies are H_0^-1 H_i (trainer.py), so video 0 is the identity at
        # every t.
        corners = torch.stack(self.corner_trajectory, dim=1)[1:]  # (N-1, video_len, 4, 2)
        acc = corners[:, 2:] - 2 * corners[:, 1:-1] + corners[:, :-2]  # (N-1, video_len-2, 4, 2)
        acc_norm = torch.norm(acc, dim=-1)  # (N-1, video_len-2, 4)
        per_video = torch.mean(acc_norm**2, dim=[1, 2])  # (N-1,)
        return per_video.mean().item()
