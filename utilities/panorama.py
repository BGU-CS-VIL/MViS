import numpy as np
import cv2
import torch
import matplotlib.cm as cm
import os
from pathlib import Path

RAINBOW_CMAP = cm.get_cmap("gist_rainbow")


def _get_color(index, total):
    """Get an RGB color tuple (0-255) from the gist_rainbow colormap."""
    t = index / max(total - 1, 1)
    rgba = RAINBOW_CMAP(t)
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def _get_warped_alpha(image_shape, adjusted_theta_np, canvas_width, canvas_height):
    """
    Create a warped alpha mask for an image given its adjusted homography.

    Parameters:
    - image_shape: (H, W) tuple of the source image
    - adjusted_theta_np: 3x3 homography matrix as numpy float32
    - canvas_width, canvas_height: canvas dimensions

    Returns:
    - warped_alpha: uint8 mask on the canvas
    """
    h, w = image_shape[:2]
    alpha = np.ones((h, w), dtype=np.uint8) * 255
    warped_alpha = cv2.warpPerspective(
        alpha, adjusted_theta_np,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    kernel = np.ones((3, 3), np.uint8)
    warped_alpha = cv2.dilate(warped_alpha, kernel, iterations=1)
    return warped_alpha


class PanoramaVideoCreator:
    """
    Class for creating panorama videos with dynamic canvas sizing based on homographies.
    """

    def __init__(self, fps=30, codec='mp4v'):
        self.fps = fps
        self.codec = codec

    def calculate_dynamic_canvas_size(self, frames, homographies, padding=100):
        """
        Calculate the required canvas size to fit all warped frames based on homographies.

        Parameters:
        - frames: List/tensor of frames with shape [num_frames, H, W, C] or [num_frames, C, H, W]
        - homographies: Tensor of homographies with shape [num_frames, 3, 3]
        - padding: Extra padding around the computed bounding box

        Returns:
        - canvas_width, canvas_height: Dimensions of the canvas
        - x_offset, y_offset: Offsets to center the panorama
        """
        all_corners = []

        # Get frame dimensions
        if isinstance(frames, torch.Tensor):
            if frames.shape[1] == 3:  # [num_frames, C, H, W]
                H, W = frames.shape[2], frames.shape[3]
            else:  # [num_frames, H, W, C]
                H, W = frames.shape[1], frames.shape[2]
        else:
            H, W = frames[0].shape[:2]

        # Process each homography to find transformed corners
        for theta in homographies:
            # Define corners in homogeneous coordinates
            corners = torch.tensor([
                [0, 0, 1],       # Top-left
                [W, 0, 1],       # Top-right
                [W, H, 1],       # Bottom-right
                [0, H, 1]        # Bottom-left
            ], dtype=torch.float32).T  # Shape: (3, 4)

            # Move corners to the same device as theta
            if theta.is_cuda:
                corners = corners.cuda()

            # Apply homography
            transformed_corners = torch.matmul(theta, corners)  # Shape: (3, 4)

            # Normalize homogeneous coordinates
            transformed_corners = transformed_corners[:2] / transformed_corners[2:]  # Shape: (2, 4)
            transformed_corners = transformed_corners.T  # Shape: (4, 2)

            all_corners.append(transformed_corners)

        all_corners = torch.cat(all_corners, dim=0)  # Shape: (4*num_frames, 2)

        min_xy = torch.floor(torch.min(all_corners, dim=0).values)
        max_xy = torch.ceil(torch.max(all_corners, dim=0).values)

        x_min, y_min = min_xy[0].item(), min_xy[1].item()
        x_max, y_max = max_xy[0].item(), max_xy[1].item()

        # Calculate canvas size with padding
        canvas_width = int(x_max - x_min + 2 * padding)
        canvas_height = int(y_max - y_min + 2 * padding)

        # Calculate offsets to center content
        x_offset = padding - x_min
        y_offset = padding - y_min

        return canvas_width, canvas_height, x_offset, y_offset

    def _draw_outlines_on_frame(self, panorama, images, thetas, canvas_width, canvas_height, x_offset, y_offset, thickness=12):
        """
        Draw rainbow-colored outlines on a panorama frame for each image's warped boundary.
        Uses offset-based translation (for video creator).

        Parameters:
        - panorama: The blended panorama frame (modified in-place)
        - images: List of images for the current timestep
        - thetas: Homography matrices for the current timestep
        - canvas_width, canvas_height: canvas dimensions
        - x_offset, y_offset: translation offsets
        - thickness: outline thickness in pixels
        """
        for i, (image, theta) in enumerate(zip(images, thetas)):
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # [C, H, W]
                    img_height, img_width = image.shape[1], image.shape[2]
                else:
                    img_height, img_width = image.shape[0], image.shape[1]
            else:
                img_height, img_width = image.shape[:2]

            translation = torch.tensor([
                [1, 0, x_offset],
                [0, 1, y_offset],
                [0, 0, 1]
            ], dtype=torch.float32, device=theta.device)

            adjusted_theta = torch.matmul(translation, theta)
            adjusted_theta_np = adjusted_theta.cpu().numpy().astype(np.float32)

            warped_alpha = _get_warped_alpha(
                (img_height, img_width), adjusted_theta_np, canvas_width, canvas_height
            )

            contours, _ = cv2.findContours(warped_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = _get_color(i, len(images))
            cv2.drawContours(panorama, contours, -1, color, thickness)

    def create_panorama_frame(self, images, thetas, canvas_width, canvas_height, x_offset, y_offset, colored_panorama=False):
        """
        Create a single panorama frame by warping and blending multiple images.

        Parameters:
        - images: List of images for the current time step
        - thetas: Homography matrices for the current time step
        - canvas_width, canvas_height: Canvas dimensions
        - x_offset, y_offset: Offsets for centering
        - colored_panorama: If True, draw rainbow-colored outlines

        Returns:
        - panorama: Blended panorama frame
        """
        # Initialize accumulator and counter
        accumulated_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        pixel_count = np.zeros((canvas_height, canvas_width), dtype=np.float32)

        for image, theta in zip(images, thetas):
            # Convert image to numpy if necessary
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # [C, H, W]
                    image = image.permute(1, 2, 0).cpu().numpy()
                else:  # Already [H, W, C]
                    image = image.cpu().numpy()
                image = (image * 255).clip(0, 255).astype(np.uint8)

            # Apply offset translation to homography
            translation = torch.tensor([
                [1, 0, x_offset],
                [0, 1, y_offset],
                [0, 0, 1]
            ], dtype=torch.float32, device=theta.device)

            adjusted_theta = torch.matmul(translation, theta)
            adjusted_theta_np = adjusted_theta.cpu().numpy().astype(np.float32)

            # Warp image
            warped_image = cv2.warpPerspective(
                image, adjusted_theta_np,
                (canvas_width, canvas_height),
                borderMode=cv2.BORDER_REPLICATE
            )

            # Create mask for valid pixels
            h, w = image.shape[:2]
            alpha = np.ones((h, w), dtype=np.uint8) * 255
            warped_alpha = cv2.warpPerspective(
                alpha, adjusted_theta_np,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Accumulate valid pixels
            valid_pixels = warped_alpha > 0
            accumulated_image[valid_pixels] += warped_image[valid_pixels].astype(np.float32)
            pixel_count[valid_pixels] += 1

        # Average the accumulated pixels
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        valid_mask = pixel_count > 0
        panorama[valid_mask] = (accumulated_image[valid_mask] /
                               pixel_count[valid_mask, np.newaxis]).clip(0, 255).astype(np.uint8)

        if colored_panorama:
            self._draw_outlines_on_frame(
                panorama, images, thetas,
                canvas_width, canvas_height, x_offset, y_offset
            )

        return panorama

    def create_panorama_video(self, frames, homographies, output_path, video_name="panorama", colored_panorama=False):
        """
        Create a panorama video directly from frames and homographies.

        Parameters:
        - frames: Tensor of frames [num_videos, num_frames, C, H, W] or [num_videos, num_frames, H, W, C]
        - homographies: Tensor of homographies [num_videos, num_frames, 3, 3]
        - output_path: Directory to save the video
        - video_name: Name of the output video file
        - colored_panorama: If True, draw rainbow-colored outlines

        Returns:
        - video_path: Path to the created video
        """
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        video_path = os.path.join(output_path, f"{video_name}.mp4")

        num_frames = frames.shape[1]

        # Calculate dynamic canvas size based on all homographies
        print(f"Calculating optimal canvas size for {num_frames} frames...")
        all_homographies = homographies.reshape(-1, 3, 3)
        canvas_width, canvas_height, x_offset, y_offset = self.calculate_dynamic_canvas_size(
            frames.reshape(-1, *frames.shape[2:]),
            all_homographies,
            padding=2
        )

        print(f"Canvas size: {canvas_width}x{canvas_height}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        video_writer = cv2.VideoWriter(
            video_path, fourcc, self.fps,
            (canvas_width, canvas_height)
        )

        if not video_writer.isOpened():
            raise ValueError(f"Failed to open video writer for {video_path}")

        # Process each frame
        print(f"Creating panorama video with {num_frames} frames...")
        for t in range(num_frames-1):
            # Get frames and homographies for current timestep
            current_frames = frames[:, t]  # [num_videos, C, H, W] or [num_videos, H, W, C]
            current_homographies = homographies[:, t]  # [num_videos, 3, 3]

            # Create panorama for current timestep
            panorama_frame = self.create_panorama_frame(
                current_frames, current_homographies,
                canvas_width, canvas_height, x_offset, y_offset,
                colored_panorama=colored_panorama
            )

            # Convert RGB to BGR for OpenCV
            panorama_frame_bgr = cv2.cvtColor(panorama_frame, cv2.COLOR_RGB2BGR)

            # Write frame to video
            video_writer.write(panorama_frame_bgr)

            # Progress indicator
            if (t + 1) % 10 == 0:
                print(f"  Processed {t + 1}/{num_frames} frames")

        # Release video writer
        video_writer.release()
        print(f"Video saved to: {video_path}")

        return video_path


def create_panorama_video_direct(frames, homographies, output_dir, video_name,
                                fps=30, colored_panorama=False):
    """
    Convenience function to create panorama video directly without saving frames.

    Parameters:
    - frames: Input frames tensor
    - homographies: Homography matrices
    - output_dir: Output directory for the video
    - video_name: Name of the output video
    - fps: Frames per second
    - colored_panorama: If True, draw rainbow-colored outlines
    """
    creator = PanoramaVideoCreator(fps=fps)
    video_path = creator.create_panorama_video(
        frames, homographies, output_dir, video_name,
        colored_panorama=colored_panorama
    )

    return video_path
