import torch
import torch.nn as nn

from models.image_warper import HomographyWarper


class AlignmentNetwork(nn.Module):
    """
    The alignment network.
    Predicts, for every frame, a zero-trace 3x3 matrix theta in sl(3).
    The homography of the frame is exp(theta), in SL(3) (Appendix B).
    """

    def __init__(self, num_videos, batch_size, img_size):
        super().__init__()
        in_channels = 3
        self.num_videos = num_videos
        self.img_size = img_size
        self.batch_size = batch_size

        self.in_norm = nn.BatchNorm2d(in_channels, affine=True, momentum=0.01)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=7),
            nn.BatchNorm2d(10),
            nn.AdaptiveMaxPool2d((32, 32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 5, kernel_size=5),
            nn.BatchNorm2d(5),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(5 * 8 * 8 * self.num_videos, 32)
        self.fc2 = nn.Linear(32, self.num_videos * 3 * 3)
        self.gelu = nn.GELU()

        # zero init: the initial homographies are the identity
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()

    def set_zero_trace_(self, thetas: torch.Tensor) -> torch.Tensor:
        """In place. thetas: [M, 3, 3], the matrices before the matrix exponential."""
        thetas[:, 2, 2] = -(thetas[:, 0, 0] + thetas[:, 1, 1])
        return thetas

    def forward(self, input: torch.Tensor) -> HomographyWarper:
        """
        input: [batch_size*t_window*num_videos, 3, res, res], frames ordered (window, time, video)
        """
        assert len(input.size()) == 4, "img must be of shape [batch_size*t_window*num_videos, 3, res, res]"

        x = self.in_norm(input)
        x = self.cnn(x)  # [batch_size*t_window*num_videos, 5, 8, 8]
        x = self.fc1(x.reshape(-1, self.num_videos * 5 * 8 * 8))
        x = self.gelu(x)
        thetas = self.fc2(x)
        thetas = thetas.reshape(-1, 3, 3)
        thetas = self.set_zero_trace_(thetas)

        return HomographyWarper(self.img_size, thetas, num_videos=self.num_videos, batch_size=self.batch_size)
