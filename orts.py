"""ORTs (Sec. 3.1): homographies between frame pairs, estimated by an off-the-shelf matcher and RANSAC.

A matcher returns keypoint matches between two frames; 
OrtEstimator fits the homographies and assembles the multi-view and temporal ORT tensors. 
A zero matrix marks a pair without an ORT.
"""
import torch
import cv2
import numpy as np
import kornia.feature as KF

from utilities.preprocess_utils import homography_to_square, scale_to_det_1
from third_party.RoMa import roma_outdoor 

class RoMaMatcher:
    """RoMa. Matches live in the height x width frame of RoMa's output resolution."""

    def __init__(self, device):
        self.device = device
        self.matcher = roma_outdoor(device=device, coarse_res=448)
        self.height, self.width = self.matcher.get_output_resolution()
        self.num_matches = 1000

    def match(self, path_a, path_b):
        """Samples num_matches matches from RoMa's dense warp.

        Returns:
            kpts_a, kpts_b: [num_matches, 2] float32 numpy arrays, (x, y) pixels in the height x width frame.
            certainty: [num_matches] RoMa certainty of each match.
        """
        warp, certainty = self.matcher.match(path_a, path_b, device=self.device)
        matches, certainty = self.matcher.sample(warp, certainty, num=self.num_matches)
        kpts_a, kpts_b = self.matcher.to_pixel_coordinates(matches, self.height, self.width, self.height, self.width)

        kpts_a = kpts_a.cpu().numpy().astype(np.float32)
        kpts_b = kpts_b.cpu().numpy().astype(np.float32)
        return kpts_a, kpts_b, certainty


class LoFTRMatcher:
    """LoFTR (kornia, outdoor weights), run on grayscale frames resized to its recommended outdoor resolution."""

    # Both dims must be multiples of 8.
    HEIGHT = 480
    WIDTH = 640

    def __init__(self, device):
        self.device = device
        self.matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
        self.height, self.width = self.HEIGHT, self.WIDTH
        self.num_matches = 2000

    def _load_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"LoFTR could not read image at {path}")
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).float()[None, None] / 255.0
        return img.to(self.device)

    def match(self, path_a, path_b):
        """Keeps the num_matches most confident LoFTR matches.

        Returns:
            kpts_a, kpts_b: [M, 2] float32 numpy arrays, (x, y) pixels in the height x width frame.
            certainty: [M] LoFTR confidence of each match.
        """
        img_a = self._load_gray(path_a)
        img_b = self._load_gray(path_b)

        with torch.no_grad():
            out = self.matcher({"image0": img_a, "image1": img_b})

        kpts_a = out["keypoints0"]
        kpts_b = out["keypoints1"]
        certainty = out["confidence"]

        if self.num_matches is not None and kpts_a.shape[0] > self.num_matches:
            topk = torch.topk(certainty, k=self.num_matches).indices
            kpts_a = kpts_a[topk]
            kpts_b = kpts_b[topk]
            certainty = certainty[topk]

        kpts_a = kpts_a.detach().cpu().numpy().astype(np.float32)
        kpts_b = kpts_b.detach().cpu().numpy().astype(np.float32)
        return kpts_a, kpts_b, certainty


class OrtEstimator:
    def __init__(self, matcher, img_res, certainty_threshold=0.3):
        """
        Args:
            matcher: RoMaMatcher or LoFTRMatcher.
            img_res: r, the side of the square frames the network sees.
            certainty_threshold: a multi-view ORT is kept only if the mean certainty of the pair's matches
                exceeds it.
        """
        self.matcher = matcher
        self.img_res = img_res
        self.certainty_threshold = certainty_threshold
        self.cx = (matcher.width - 1.0) / 2.0
        self.cy = (matcher.height - 1.0) / 2.0

        # [0, width-1] x [0, height-1] pixels (origin at the top-left corner) -> [-1, 1]^2 (origin at the image centre)
        self.T_norm = np.array([[1.0/self.cx, 0.0,         -1.0],
                                [0.0,         1.0/self.cy, -1.0],
                                [0.0,         0.0,          1.0]], dtype=np.float32)
        # and back
        self.T_norm_inv = np.array([[self.cx, 0.0,     self.cx],
                                    [0.0,     self.cy, self.cy],
                                    [0.0,     0.0,     1.0]], dtype=np.float32)

    def estimate_homographies(self, kpts_a, kpts_b):
        """Fits the homography that maps frame a to frame b with RANSAC.

        Args:
            kpts_a, kpts_b: [M, 2] matched pixels in the matcher's height x width frame.

        Returns:
            H_hat_a_to_b: [3, 3] float32, the ORT: normalized coordinates ([-1, 1]^2, origin at the image
                centre), scaled to det 1.
            H_hat_a_to_b_square: [3, 3] float32, the same map in pixels of the r x r frames, scaled to det 1.
                Used only for the overlap masks.
            Both are None if RANSAC finds no homography.
        """
        H_pix, _ = cv2.findHomography(kpts_a, kpts_b, cv2.RANSAC, 5.0)
        if H_pix is None:
            return None, None

        H_hat_a_to_b = torch.tensor(scale_to_det_1(self.T_norm @ H_pix @ self.T_norm_inv), dtype=torch.float32)

        H_pix_det1 = torch.tensor(scale_to_det_1(H_pix), dtype=torch.float32)
        H_hat_a_to_b_square = homography_to_square(H_pix_det1, self.matcher.height, self.matcher.width, self.img_res)

        return H_hat_a_to_b, H_hat_a_to_b_square

    def estimate_pair(self, path_a, path_b):
        """Matches frames a and b once and fits the homographies in both directions.

        Returns:
            (H_hat_a_to_b, H_hat_a_to_b_square, H_hat_b_to_a, H_hat_b_to_a_square, certainty_mean),
            or None if there are fewer than 4 matches or RANSAC fails in either direction.
        """
        kpts_a, kpts_b, certainty = self.matcher.match(path_a, path_b)
        if len(kpts_a) < 4:
            return None

        H_hat_a_to_b, H_hat_a_to_b_square = self.estimate_homographies(kpts_a, kpts_b)
        H_hat_b_to_a, H_hat_b_to_a_square = self.estimate_homographies(kpts_b, kpts_a)
        if H_hat_a_to_b is None or H_hat_b_to_a is None:
            return None

        return H_hat_a_to_b, H_hat_a_to_b_square, H_hat_b_to_a, H_hat_b_to_a_square, certainty.mean()

    def create_mv_orts(self, imgs_paths, t):
        """
        Args:
            imgs_paths: [N] paths of the N videos' frames at time t.
            t: frame index, used only in the log.

        Returns:
            orts: [N, N, 3, 3], orts[i, j] is the ORT from frame i to frame j. Zero on the diagonal and where
                there is no ORT.
            orts_square: [N, N, 3, 3], the same in r x r pixels.
        """
        N = len(imgs_paths)
        orts = torch.zeros((N, N, 3, 3), device=self.matcher.device)
        orts_square = torch.zeros((N, N, 3, 3), device=self.matcher.device)
        for i in range(N):
            for j in range(N):
                if i < j:
                    pair = self.estimate_pair(imgs_paths[i], imgs_paths[j])
                    if pair is None:
                        print(f"t={t}: no homography between videos {i} and {j} "
                              f"(fewer than 4 matches or RANSAC failed), ORT set to zero")
                        continue
                    H_hat_i_to_j, H_hat_i_to_j_square, H_hat_j_to_i, H_hat_j_to_i_square, certainty_mean = pair

                    if certainty_mean > self.certainty_threshold:
                        orts[i, j] = H_hat_i_to_j
                        orts_square[i, j] = H_hat_i_to_j_square
                        orts[j, i] = H_hat_j_to_i
                        orts_square[j, i] = H_hat_j_to_i_square
                    else:
                        print(f"t={t}: no overlap between videos {i} and {j} "
                              f"(mean certainty {float(certainty_mean):.3f} <= {self.certainty_threshold}), ORT set to zero")

        return orts, orts_square

    def create_temporal_orts(self, imgs_paths, t):
        """
        Args:
            imgs_paths: [N, 2] paths of each video's frames at times t and t+1.
            t: frame index, used only in the log.

        Returns:
            orts: [2, N, 3, 3], orts[0, i] is the ORT of video i from t to t+1 and orts[1, i] from t+1 to t.
                Zero only where no homography could be fitted.
            orts_square: [2, N, 3, 3], the same in r x r pixels.
        """
        N = len(imgs_paths)
        orts = torch.zeros((2, N, 3, 3), device=self.matcher.device)
        orts_square = torch.zeros((2, N, 3, 3), device=self.matcher.device)
        for i in range(N):
            pair = self.estimate_pair(imgs_paths[i][0], imgs_paths[i][1])
            if pair is None:
                print(f"video {i}: no homography between t={t} and t={t + 1} "
                      f"(fewer than 4 matches or RANSAC failed), ORT set to zero")
                continue
            H_hat_t_to_tp1, H_hat_t_to_tp1_square, H_hat_tp1_to_t, H_hat_tp1_to_t_square, _ = pair

            orts[0, i] = H_hat_t_to_tp1
            orts_square[0, i] = H_hat_t_to_tp1_square
            orts[1, i] = H_hat_tp1_to_t
            orts_square[1, i] = H_hat_tp1_to_t_square

        return orts, orts_square
