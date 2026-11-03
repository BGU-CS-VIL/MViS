import argparse

# The default --batch_size of every dataset
DATASET_BATCH_SIZES = {'new_scene': 4, 'MMCD': 4, 'StabStitch-D': 6}


def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Data
    parser.add_argument('--dataset', type=str, default='new_scene', choices=['new_scene', 'MMCD', 'StabStitch-D'],
                        help='new_scene: the scene in --data_folder')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='new_scene: folder of the scene, with video0, video1, ... (required). '
                             'MMCD: folder with frames/<scene>/video<i> and kps_annotations/<scene> (default: ../MMCD). '
                             'StabStitch-D: folder with <scene>/video<i> (default: ../stabstitch-d)')
    parser.add_argument('--scene', type=str, default=None,
                        help='MMCD, StabStitch-D: run only this scene. Default: all of the scenes of the dataset')
    parser.add_argument('--image_resolution', type=int, default=256,
                        help='r: the frames are resized to r x r before they are fed to the network')

    # Preprocessing
    parser.add_argument('--matcher', type=str, default='RoMa', choices=['RoMa', 'LoFTR'],
                        help='Matcher used to estimate the ORTs')
    parser.add_argument('--certainty_threshold', type=float, default=0.3,
                        help='Minimum mean match certainty (RoMa certainty, LoFTR confidence) required to accept an ORT')

    # Stage 1: geometric loss
    parser.add_argument('--stage1_epochs', type=int, default=600,
                        help='Number of epochs')
    parser.add_argument('--stage1_lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--stage1_patience', type=int, default=50,
                        help='Number of epochs ReduceLROnPlateau waits before reducing the learning rate')
    parser.add_argument('--stage1_d', type=int, default=7,
                        help='d: the geometric loss compares transformations on a d x d grid G (paper Sec. 3.3)')

    # Stage 2: appearance loss
    parser.add_argument('--stage2_epochs', type=int, default=600,
                        help='Number of epochs')
    parser.add_argument('--stage2_lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--stage2_patience', type=int, default=50,
                        help='Number of epochs ReduceLROnPlateau waits before reducing the learning rate')

    # Both stages
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of sliding windows in a batch, in both stages. '
                             'Default: 4 on new_scene and MMCD, 6 on StabStitch-D')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True,
                        help='Shuffle the sliding windows in every epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay of Adam')

    # Panoramic video
    parser.add_argument('--results_folder', type=str, default='results',
                        help='Folder in which the panoramic videos are saved')
    parser.add_argument('--panoramic_video_type', type=str, default='none',
                        choices=['moving', 'fixed', 'both', 'none'],
                        help='Which panoramic video(s) to save after evaluation. Since the predicted homographies '
                             'are defined only up to a global transformation, a reference is chosen for visualization '
                             '(paper Sec. 3.6). '
                             'fixed: at each time t, the inverse of the homography of video 0 at time t is '
                             'applied to all frames at that time, so video 0 stays fixed in the panorama and the '
                             'other videos move relative to it. '
                             'moving: the inverse of the homography of the first frame of video 0 is applied to all '
                             'frames across all times and views, so all videos, including video 0, move in one '
                             'shared panoramic coordinate system. '
                             'both: save the fixed and the moving video. '
                             'none: do not save a panoramic video')
    parser.add_argument('--colored_panorama', action=argparse.BooleanOptionalAction, default=False,
                        help='Draw rainbow-colored outlines around each input image in the panorama')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame rate of the saved panoramic videos')

    return parser
