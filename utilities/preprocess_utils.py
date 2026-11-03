import torch
from PIL import Image
from torchvision import transforms
import random
import os
import numpy as np
from typing import List, Tuple
from pathlib import Path
import cv2

# Used in: extract_frames (this file).
def get_image_files(folder: Path) -> List[Path]:
    """Return a sorted list of image files in the folder."""
    return sorted(
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.png"))
    )
    
# Used in: extract_frames (this file).
def load_and_transform(file: Path, device: torch.device, square_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load an image from file, convert it to a tensor, and also return a square (resized) version.
    
    Args:
        file: Path to the image file.
        device: The torch device to move the tensor to.
        square_size: The desired resolution for the square image.
    
    Returns:
        A tuple (original_tensor, square_tensor).
    """
    
    im_pil = Image.open(file).convert("RGB")
    im_tensor = transforms.ToTensor()(im_pil).to(device)
    # Resize the image to the desired square resolution.
    im_square = transforms.ToTensor()(im_pil.copy().resize((square_size, square_size))).to(device)
    return im_tensor, im_square

    
# Used in: demo.py (main), trainer.py (get_num_videos).
def count_videos(scene_folder: Path) -> int:
    """The number of videos N: the folders video0, video1, ... in scene_folder, counted up to the first missing one."""
    num_videos = 0
    while (scene_folder / f'video{num_videos}').is_dir():
        num_videos += 1
    if num_videos < 2:
        raise ValueError(f'{scene_folder} must contain at least two video folders (video0, video1, ...), '
                         f'found {num_videos}')
    return num_videos

# Used in: preprocessor.py (Preprocessor.extract_frames, Preprocessor.preprocess).
def extract_frames(data_folder: Path, num_videos: int, img_resolution: int, device) -> Tuple:
    """
    Extract frames from video folders and return both the original and resized (square) versions.
    
    Assumes a folder structure with a 'videos' subfolder containing directories named 'video0', 'video1', etc.
    
    Args:
        data_folder: Base path containing the 'videos' folder.
        num_videos: Number of video folders to process.
        img_resolution: Resolution for the square images.
        device: The torch device to use.
    
    Returns:
        A tuple (frames_orgs, frames_square):
            - frames_orgs: A tensor of shape (num_videos, num_frames, 3, h, w)
            - frames_square: A tensor of shape (num_videos, num_frames, 3, img_resolution, img_resolution)
            - files_arr: A numpy array of shape (num_videos, num_frames) containing the paths to the images
    """
    videos_folder = data_folder
    
    # Process the first video to determine the number of frames.
    video0_folder = videos_folder / "video0"
    files = get_image_files(video0_folder)
    video_len = len(files)
    # Preallocate tensor for square images.
    frames_square = torch.zeros((num_videos, video_len, 3, img_resolution, img_resolution), device=device)
    
    files_lists_of_lists = [] 
    
    # For uniform input shapes, determine dimensions from the first frame of video0.
    first_frame, _ = load_and_transform(files[0], device, img_resolution)
    h, w = first_frame.shape[1:]
    frames_orgs = torch.zeros((num_videos, video_len, 3, h, w), device=device)
    
    for i in range(num_videos):
        video_folder = videos_folder / f"video{i}"
        files = get_image_files(video_folder)
        files_lists_of_lists.append(files)
        for j in range(video_len):
            frame, frame_square = load_and_transform(files[j], device, img_resolution)
            frames_orgs[i, j] = frame
            frames_square[i, j] = frame_square
    files_arr = np.array(files_lists_of_lists)                 
    return frames_orgs, frames_square, files_arr

# Used in: orts.py (OrtEstimator.estimate_homographies).
def scale_to_det_1(H):
    """Divides a homography by the cube root of its determinant. The map it represents is unchanged."""
    det_H = np.linalg.det(H)
    if det_H == 0:
        raise ValueError("Homography matrix has a determinant of 0; cannot normalize.")
    return H / np.cbrt(det_H)

# Used in: orts.py (OrtEstimator.estimate_homographies).
def homography_to_square(H, height, width, square_size):
    """Expresses a homography between two height x width images in pixels of the same images resized to
    square_size x square_size.

    Args:
        H: [3, 3] float32 tensor, maps (x, y, 1) pixels of image a to pixels of image b (both height x width).
        height, width: size of the images H was estimated on.
        square_size: r, side of the square frames.

    Returns:
        [3, 3] float32 tensor S @ H @ S^-1, with S = diag(r / width, r / height, 1) the resize of the pixel
        coordinates.
    """
    S = torch.tensor([[square_size / width, 0, 0],
                      [0, square_size / height, 0],
                      [0, 0, 1.0]], dtype=torch.float32, device=H.device)
    return S @ H @ torch.linalg.inv(S)

# Used in: train.py (main).
def seed_everything(seed):
    if seed is not None and seed >= 0:
        print(f"Setting seed to {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True   
        torch.backends.cudnn.benchmark = False  
    
        # OpenCV
        cv2.setRNGSeed(seed)
    else:
        print("No seed set, results will not be reproducible")
