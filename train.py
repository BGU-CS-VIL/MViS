"""Runs MViS on one scene or on a benchmark.
"""
from copy import deepcopy
from pathlib import Path

import torch

from train_args import DATASET_BATCH_SIZES, get_argparser
from trainer import Trainer
from utilities.preprocess_utils import seed_everything

# The categories are run in this order
STABSTITCH_D_CATEGORIES = ['FM', 'RE', 'LL', 'LT']
STABSTITCH_D_SCENES = {'FM': ['168', '175', '224', 'MR006', 'SF34'],
                       'LL': ['100', '99', '90', '85', '74'],
                       'LT': ['21', '37', '40', '140', 'ML001'],
                       'RE': ['107', '101', 'MR002', 'S13', 'S28']}
# Number of frames of every scene in STABSTITCH_D_SCENES: its weight in the average over the frames
STABSTITCH_D_NUM_FRAMES = {'FM': [124, 313, 190, 459, 180],
                           'LL': [445, 958, 334, 718, 238],
                           'LT': [598, 682, 382, 535, 423],
                           'RE': [817, 208, 270, 581, 786]}
# The MMCD scenes of the paper, by number of videos N
MMCD_SCENES = {3: ['scene_1', 'scene_2', 'scene_4', 'scene_6', 'scene_7', 'scene_8', 'scene_9', 'scene_10',
                   'scene_12', 'scene_13', 'scene_15', 'scene_18', 'scene_19', 'scene_20', 'scene_21', 'scene_29',
                   'scene_39', 'scene_67', 'scene_69', 'scene_70', 'scene_106', 'scene_110'],
               4: ['scene_25', 'scene_30', 'scene_31', 'scene_35', 'scene_47', 'scene_54', 'scene_64', 'scene_49',
                   'scene_92', 'scene_81'],
               5: ['scene_33', 'scene_46', 'scene_51', 'scene_62', 'scene_63', 'scene_65', 'scene_66', 'scene_72',
                   'scene_76', 'scene_77', 'scene_98'],
               6: ['scene_27', 'scene_50', 'scene_52', 'scene_53', 'scene_55', 'scene_57', 'scene_58', 'scene_59',
                   'scene_60', 'scene_101']}
# The default --data_folder of every dataset
DATASET_FOLDERS = {'MMCD': '../MMCD', 'StabStitch-D': '../stabstitch-d'}
SEPARATOR = '-' * 95


def parse_args():
    """Parses the command line (train_args.py) and returns the arguments as a dict."""
    parser = get_argparser()
    args = vars(parser.parse_args())

    if args['dataset'] == 'new_scene':
        if args['data_folder'] is None:
            parser.error('--dataset new_scene requires --data_folder')
    elif args['data_folder'] is None:
        args['data_folder'] = DATASET_FOLDERS[args['dataset']]
    if args['batch_size'] is None:
        args['batch_size'] = DATASET_BATCH_SIZES[args['dataset']]
    return args


def scene_args(args, scene):
    """Returns a copy of args for one scene of the dataset, with 'scene' and 'scene_folder' set."""
    frames_folder = Path(args['data_folder'])
    if args['dataset'] == 'MMCD':
        frames_folder = frames_folder / 'frames'
    args = deepcopy(args)
    args['scene'] = scene
    args['scene_folder'] = str(frames_folder / scene)
    return args


def run_stabstitch_d_category(category_name, args):
    """Runs MViS on the scenes of one StabStitch-D category, one after the other.

    Args:
        category_name: 'FM', 'LL', 'LT' or 'RE'.

    Returns:
        (psnr_sum, ssim_sum, num_frames)
    """
    category_scene_names = STABSTITCH_D_SCENES[category_name]
    category_scene_num_frames = STABSTITCH_D_NUM_FRAMES[category_name]

    num_frames = sum(category_scene_num_frames)
    psnr_sum = 0.0
    ssim_sum = 0.0

    for i, scene in enumerate(category_scene_names):
        print(f'Training on data: {scene}')
        trainer = Trainer(scene_args(args, scene))
        psnr, ssim, _, _, _ = trainer.train()
        print(SEPARATOR)
        print(f'Finished training on data: {scene}, psnr: {psnr}, ssim: {ssim}')
        print(SEPARATOR)
        psnr_sum += psnr * category_scene_num_frames[i]
        ssim_sum += ssim * category_scene_num_frames[i]
        del trainer
        torch.cuda.empty_cache()
    category_ssim = ssim_sum / num_frames
    category_psnr = psnr_sum / num_frames
    print(SEPARATOR)
    print(f'{category_name} Final psnr: {category_psnr}, ssim: {category_ssim}')
    print(SEPARATOR)
    print('\n')
    torch.cuda.empty_cache()
    return psnr_sum, ssim_sum, num_frames


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device} ({device_name})')

    args = parse_args()
    args['device'] = device
    dataset = args['dataset']
    seed_everything(args['seed'])

    if dataset == 'new_scene':
        scene_folder = Path(args['data_folder'])
        args['scene'] = scene_folder.resolve().name
        args['scene_folder'] = str(scene_folder)
        trainer = Trainer(args)
        trainer.train()
        print(SEPARATOR)
        print(f"Finished training on data: {args['scene']}")
        print(SEPARATOR)

    elif args['scene'] is not None:
        trainer = Trainer(scene_args(args, args['scene']))
        psnr, ssim, gv, stability, cmsa = trainer.train()
        print(SEPARATOR)
        if dataset == 'MMCD':
            print(f"Finished training on data: {args['scene']}, geo_var: {gv}, stability: {stability} cmsa: {cmsa}")
        else:
            print(f"Finished training on data: {args['scene']}, psnr: {psnr}, ssim: {ssim}")
        print(SEPARATOR)

    elif dataset == 'StabStitch-D':
        total_num_frames = 0.0
        total_psnr_sum = 0.0
        total_ssim_sum = 0.0
        for category_name in STABSTITCH_D_CATEGORIES:
            print(f'beginning to run category {category_name}')
            psnr_sum, ssim_sum, num_frames = run_stabstitch_d_category(category_name, args)
            total_psnr_sum = total_psnr_sum + psnr_sum
            total_ssim_sum = total_ssim_sum + ssim_sum
            total_num_frames = total_num_frames + num_frames
            torch.cuda.empty_cache()

        avg_ssim = total_ssim_sum / total_num_frames
        avg_psnr = total_psnr_sum / total_num_frames
        print(SEPARATOR)
        print(f'AVG  psnr: {avg_psnr}, AVG  ssim: {avg_ssim}')
        print(SEPARATOR)

    elif dataset == 'MMCD':
        for num_videos in range(3, 7):
            print(SEPARATOR)
            print(f'Running MViS on MMCD category N={num_videos}')
            gvs, stabilities, cmsas = [], [], []

            for scene in MMCD_SCENES[num_videos]:
                print(f'Training on data: {scene}')
                trainer = Trainer(scene_args(args, scene))
                _, _, gv, stability, cmsa = trainer.train()
                gvs.append(gv)
                stabilities.append(stability)
                cmsas.append(cmsa)
                print(SEPARATOR)
                print(f'Finished training on data: {scene}, geo_var: {gv}, stability: {stability} cmsa: {cmsa}')
                print(SEPARATOR)
                del trainer
                torch.cuda.empty_cache()
            mean_gv = sum(gvs) / len(gvs)
            mean_stability = sum(stabilities) / len(stabilities)
            mean_cmsa = sum(cmsas) / len(cmsas)
            print(SEPARATOR)
            print(f'Finished training on MMCD category N={num_videos}')
            print(f'AVG geo_var: {mean_gv}, AVG stability: {mean_stability}, AVG cmsa: {mean_cmsa}')
            print(SEPARATOR)


if __name__ == '__main__':
    main()
