"""Runs MViS on a new scene and saves its panoramic video.

The scene folder holds one folder of frames per video: video0, video1, ... (.jpg, .jpeg or .png, read in file-name
order). The videos must be synchronized and have the same number of frames and the same frame size.

    python demo.py --scene_folder path/to/scene --video both

The videos are saved to <results_folder>/<scene>/fixed_panorama_<scene>.mp4 and moving_panorama_<scene>.mp4.
"""
import argparse
from pathlib import Path

import torch

from train_args import DATASET_BATCH_SIZES, get_argparser
from trainer import Trainer
from utilities.preprocess_utils import count_videos, seed_everything


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene_folder', type=str, required=True,
                        help='Folder of the scene, with one folder of frames per video: video0, video1, ...')
    parser.add_argument('--video', type=str, default='both', choices=['fixed', 'moving', 'both'],
                        help='Which panoramic video(s) to save. '
                             'fixed: video 0 stays fixed in the panorama and the other videos move relative to it. '
                             'moving: all videos, including video 0, move in one shared panoramic coordinate system. '
                             'both: save the fixed and the moving video')
    parser.add_argument('--results_folder', type=str, default='results',
                        help='Folder in which the panoramic videos are saved')
    demo_args = parser.parse_args()

    scene_folder = Path(demo_args.scene_folder)
    num_videos = count_videos(scene_folder)
    scene_name = scene_folder.resolve().name

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    args = vars(get_argparser().parse_args([]))
    args.update(dataset='new_scene', scene_folder=str(scene_folder), scene=scene_name,
                batch_size=DATASET_BATCH_SIZES['new_scene'], panoramic_video_type=demo_args.video,
                results_folder=demo_args.results_folder, device=device)
    seed_everything(args['seed'])

    print(f'Running MViS on {scene_folder} ({num_videos} videos)')
    Trainer(args).train()
    print(f"Panoramic video(s) saved to {Path(args['results_folder']) / scene_name}")


if __name__ == '__main__':
    main()
