from collections import namedtuple

from torch import nn
from torchvision import transforms

from network import MEDFE

LAMBDA = {
    "reconstruction_out": 1,
    "reconstruction_structure": 1,
    "reconstruction_texture": 1,
    "perceptual": 0.1,
    "style": 250,
    "adversarial": 0.2
}
to_tensor = transforms.ToTensor()


class PerceptualLoss(nn.Module):
    """
    https://github.com/ceshine/fast-neural-style --> loss_network.py
    """

    def __init__(self, model: MEDFE):
        super().__init__()
        self.model = model

    def forward(self, i_gt, i_out):
        # TODO capture activation of model on both images
        loss = 0
        # self.model.relu1
        # self.model.relu2
        # self.model.relu3
        # self.model.relu4
        # self.model.relu5


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_rst = nn.L1Loss(reduction="sum")
        self.lost_rte = nn.L1Loss(reduction="sum")
        self.loss_re = nn.L1Loss(reduction="sum")

    def forward(self, i_gt, i_st, i_ost, i_ote, i_out):
        """
        Compute the total loss. All input images are RGB and 32x32.

        :param i_gt: Ground truth image (unmasked, downsampled)
        :param i_st: 'Structure Image' of i_gt
        :param i_ost: Output of structure branch, mapped to RGB using a 1x1 convolution
        :param i_ote: Output of texture branch, mapped to RGB using a 1x1 convolution
        :param i_out: Final predicted image
        :return: Scalar loss
        """
        return (LAMBDA["reconstruction_structure"] * self.loss_rst(to_tensor(i_ost), to_tensor(i_st)) +
                LAMBDA["reconstruction_texture"] * self.lost_rte(to_tensor(i_ote), to_tensor(i_gt)) +
                LAMBDA["reconstruction_out"] * self.loss_re(to_tensor(i_out), to_tensor(i_gt)))
