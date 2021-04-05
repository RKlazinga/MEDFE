import torch
from torch import nn
from torchvision import transforms
import numpy as np

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


class StylePerceptualLoss(nn.Module):
    """
    Combination of the style and perceptual loss functions.
    Since these use the same activation maps, it is more efficient to compute them together.

    Perceptual loss:
    https://github.com/ceshine/fast-neural-style --> loss_network.py
    """

    def __init__(self, model: MEDFE):
        super().__init__()
        self.model = model
        self.percept_loss1 = nn.L1Loss()
        self.percept_loss2 = nn.L1Loss()
        self.percept_loss3 = nn.L1Loss()
        self.percept_loss4 = nn.L1Loss()
        self.percept_loss5 = nn.L1Loss()

        self.style_loss1 = nn.L1Loss()
        self.style_loss2 = nn.L1Loss()
        self.style_loss3 = nn.L1Loss()
        self.style_loss4 = nn.L1Loss()
        self.style_loss5 = nn.L1Loss()

        self.all_1_mask_256 = torch.tensor(np.full((256, 256), 256).reshape(1, 1, 256, 256)).float()

    @staticmethod
    def gram_matrix(tensor):
        """
        With thanks to M. Doosti Lakhani (Nikronic):
        https://discuss.pytorch.org/t/implementation-of-gram-matrix-in-neural-style-tutorial/46803

        :param tensor: Any input tensor
        :return: The Gram Matrix of the tensor
        """

        # get the batch_size, depth, height, and width of the Tensor
        n, d, h, w = tensor.size()

        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(n, d, h * w)

        # calculate the gram matrix
        gram = torch.matmul(tensor, tensor.transpose(1, 2))

        return gram

    def _add_mask(self, tensor):
        n = tensor.shape[0]
        return torch.cat((tensor, self.all_1_mask_256.expand(n, -1, -1, -1)), dim=1)

    def forward(self, i_gt, i_out):
        """
        Calculate perceptual loss.

        :param i_gt: 3x256x256 tensor representing the unmasked source image
        :param i_out: 3x256x256 tensor, the output of the network on the masked image
        :return: A float
        """
        # TODO possibly uncomment this if everything dies on Perceptual Loss
        # state = self.model.state_dict(keep_vars=True)
        with torch.no_grad():
            n = i_gt.shape[0]
            actual_mask = self.model.set_mask(self.all_1_mask_256.expand(n, -1, -1, -1))

            self.model(self._add_mask(i_gt))
            activation_gt = (
                self.model.relu1_cache.clone(),
                self.model.relu2_cache.clone(),
                self.model.relu3_cache.clone(),
                self.model.relu4_cache.clone(),
                self.model.relu5_cache.clone()
            )

            self.model(self._add_mask(i_out))
            activation_out = (
                self.model.relu1_cache.clone(),
                self.model.relu2_cache.clone(),
                self.model.relu3_cache.clone(),
                self.model.relu4_cache.clone(),
                self.model.relu5_cache.clone()
            )

            self.model.set_mask(actual_mask)
        # self.model.load_state_dict(state)

        percept_loss = 0
        percept_loss += self.percept_loss1(activation_gt[0], activation_out[0]) / 64
        percept_loss += self.percept_loss2(activation_gt[1], activation_out[1]) / 128
        percept_loss += self.percept_loss3(activation_gt[2], activation_out[2]) / 256
        percept_loss += self.percept_loss4(activation_gt[3], activation_out[3]) / 512
        percept_loss += self.percept_loss5(activation_gt[4], activation_out[4]) / 512
        percept_loss *= LAMBDA["perceptual"]

        gram_gt = [self.gram_matrix(x) for x in activation_gt]
        gram_out = [self.gram_matrix(x) for x in activation_out]

        style_loss = 0
        style_loss += self.style_loss1(gram_gt[0], gram_out[0])
        style_loss += self.style_loss2(gram_gt[1], gram_out[1])
        style_loss += self.style_loss3(gram_gt[2], gram_out[2])
        style_loss += self.style_loss4(gram_gt[3], gram_out[3])
        style_loss += self.style_loss5(gram_gt[4], gram_out[4])
        style_loss *= LAMBDA["style"]

        return percept_loss + style_loss


class TotalLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.loss_rst = nn.L1Loss(reduction="sum")
        self.lost_rte = nn.L1Loss(reduction="sum")
        self.loss_re = nn.L1Loss(reduction="sum")
        self.style_percept_loss = StylePerceptualLoss(model)

    def forward(self, i_gt, i_st, i_ost, i_ote, i_out, i_gt_large, i_out_large):
        """
        Compute the total loss. All input images are 3x32x32 tensors unless specified otherwise.

        :param i_gt: Ground truth image (unmasked)
        :param i_st: 'Structure Image' of i_gt
        :param i_ost: Output of structure branch, mapped to RGB using a 1x1 convolution
        :param i_ote: Output of texture branch, mapped to RGB using a 1x1 convolution
        :param i_out: Final predicted image
        :param i_gt_large: Ground truth image (unmasked, 3x256x256)
        :param i_out_large: Output image (3x256x256)
        :return: Scalar loss
        """
        return (LAMBDA["reconstruction_structure"] * self.loss_rst(i_ost, self._remove_mask(i_st)) +
                LAMBDA["reconstruction_texture"] * self.lost_rte(i_ote, self._remove_mask(i_gt)) +
                LAMBDA["reconstruction_out"] * self.loss_re(i_out, self._remove_mask(i_gt)) +
                self.style_percept_loss(self._remove_mask(i_gt_large), i_out_large))

    @staticmethod
    def _remove_mask(tensor):
        c = tensor.shape[1]
        return torch.split(tensor, [c - 1, 1], dim=1)[0]
