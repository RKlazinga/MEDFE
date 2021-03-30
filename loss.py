from torch import nn
from torchvision import transforms

LAMBDA = {
    "reconstruction_out": 1,
    "reconstruction_structure": 1,
    "reconstruction_texture": 1,
    "perceptual": 0.1,
    "style": 250,
    "adversarial": 0.2
}
to_tensor = transforms.ToTensor()


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
