import torch
from torch.utils.data import Dataset as TorchDataset


# t_window = 2 : every sample is a window of 2 consecutive frames, relative times t=0,1.

# frames (both datasets): the raw video frames after the preprocessing of paper Sec. 3.1 (Preprocessor.preprocess):
#   1. resized to r x r (r = --image_resolution, 256)
#   2. scaled to [0, 1] 
#   3. RGB-normalized


class Stage1Dataset(TorchDataset):
    def __init__(self, frames, all_mv_orts, all_temporal_orts, video_len):
        self.video_len = video_len
        self.frames = frames.permute(1,0,2,3,4)     # [video_len, num_video, 3, r, r]
        self.all_mv_orts = all_mv_orts              # [video_len, num_videos, num_videos, 3, 3]
        self.all_temporal_orts = all_temporal_orts  # [video_len-1, 2, num_videos, 3, 3]
        t_idxs = torch.arange(0, video_len, dtype=torch.long)
        self.b_idxs = [t_idxs[i:i+2] for i in range(video_len-1)]


    def get_img_data(self, index):
        indices = self.b_idxs[index] # [t_window]
        frames = self.frames[indices]                             # [t_window, num_videos, 3, r, r]
        mv_orts = self.all_mv_orts[indices]                       # [t_window, num_videos, num_videos, 3, 3]
        temporal_orts = self.all_temporal_orts[indices[:-1]]         # [t_window-1, 2, num_videos, 3, 3]
        return frames, mv_orts, temporal_orts

    def __getitem__(self, index):
        frames, mv_orts, temporal_orts = self.get_img_data(index)
        sample = {"frames": frames,                                          # [t_window, num_videos, 3, r, r]
                  "mv_orts": mv_orts,                                        # [t_window, num_videos, num_videos, 3, 3]
                  "temporal_orts": temporal_orts,	                         # [t_window-1, 2, num_videos, 3, 3]
                  "t": self.b_idxs[index][0]}               # int
        return sample

    def __len__(self):
        return self.video_len - 1





class Stage2Dataset(TorchDataset):
    def __init__(self, frames, all_mv_masks, all_temporal_masks, video_len):
        self.video_len = video_len
        self.frames = frames.permute(1,0,2,3,4)       # [video_len, num_video, 3, r, r]
        self.all_mv_masks = all_mv_masks              # [video_len, num_videos, num_videos, r, r]
        self.all_temporal_masks = all_temporal_masks  # [video_len-1, 2, num_videos, r, r]
        t_idxs = torch.arange(0, video_len, dtype=torch.long)
        self.b_idxs = [t_idxs[i:i+2] for i in range(video_len-1)]

    def get_img_data(self, index):
        indices = self.b_idxs[index]                              # [t_window]
        frames = self.frames[indices]                             # [t_window, num_videos, 3, r, r]
        mv_masks = self.all_mv_masks[indices]                     # [t_window, num_videos, num_videos, r, r]
        temporal_masks = self.all_temporal_masks[indices[:-1]]    # [t_window-1, 2, num_videos, r, r]

        return frames, mv_masks, temporal_masks


    def __getitem__(self, index):
        frames, mv_masks, temporal_masks = self.get_img_data(index)
        st = self.b_idxs[index][0]
        sample = {"frames": frames,                                          # [t_window, num_videos, 3, r, r]
                "mv_masks": mv_masks,                                       # [t_window, num_videos, num_videos, r, r]
                "temporal_masks": temporal_masks,	                         # [t_window-1, 2, num_videos, r, r]
                "t": st}               # int
        return sample

    def __len__(self):
        return self.video_len - 1
