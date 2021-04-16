import torch
from torch import nn
from torchvision import transforms
import torchvision.models.vgg as vgg

from training.wgan import Discriminator

LAMBDA = {
    "reconstruction_out": 1,
    "reconstruction_structure": 1,
    "reconstruction_texture": 1,
    "perceptual": 0.1,
    "style": 250,
    "adversarial": 0.2
}

ENABLED = {
    "reconstruction_out": True,
    "reconstruction_structure": True,
    "reconstruction_texture": True,
    "style": False,
    "perceptual": False,
    "adversarial": False,
}

to_tensor = transforms.ToTensor()


class TotalLoss(nn.Module):
    """
    Loss module that combines all loss functions used in the paper.
    These can be turned on and off using the settings above.
    """
    def __init__(self, wgan_global: Discriminator, wgan_local: Discriminator):
        super().__init__()

        self.wgan_global = wgan_global
        self.wgan_local = wgan_local

        self.loss_rst = nn.L1Loss()
        self.lost_rte = nn.L1Loss()
        self.loss_re = nn.L1Loss()
        self.style_percept_loss = StylePerceptualLoss()

        self.last_loss = {}

    def forward(self, i_gt_small, i_st, i_ost, i_ote, i_gt_large, i_out_large, i_gt_sliced, i_out_sliced, mask_size):
        """
        Compute the total loss. All input images are 3x32x32 tensors unless specified otherwise.

        :param i_gt_small: Ground truth image (unmasked)
        :param i_st: 'Structure Image' of i_gt
        :param i_ost: Output of structure branch, mapped to RGB using a 1x1 convolution
        :param i_ote: Output of texture branch, mapped to RGB using a 1x1 convolution
        :param i_gt_large: Ground truth image (unmasked, 3x256x256)
        :param i_out_large: Output image (3x256x256)
        :param i_gt_sliced: Ground trunth sliced to mask shape
        :param i_out_sliced: Output image sliced to mask shape
        :param mask_size: Number of masked pixels
        :return: Scalar loss
        """
        self.last_loss = {}

        if ENABLED['style'] or ENABLED['perceptual']:
            style_loss, percept_loss = self.style_percept_loss(i_gt_large, i_out_large)

        for loss_name, is_enabled in ENABLED.items():
            if is_enabled:
                if loss_name == "reconstruction_out":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * self.loss_re(i_out_large, i_gt_large)
                    # scale the reconstruction loss by all the pixels that were unmasked,
                    # and that should therefore not contribute to a low L1Loss
                    self.last_loss[loss_name] *= (256 * 256) / mask_size
                if loss_name == "reconstruction_texture":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * self.lost_rte(i_ote, i_gt_small)
                if loss_name == "reconstruction_structure":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * self.loss_rst(i_ost, i_st)
                if loss_name == "style":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * style_loss
                if loss_name == "perceptual":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * percept_loss
                if loss_name == "adversarial":
                    self.last_loss[loss_name] = LAMBDA[loss_name] * self.calc_adversarial(
                        i_gt_large, i_out_large, i_gt_sliced, i_out_sliced
                    )
        return sum(self.last_loss.values())

    def calc_adversarial(self, i_gt_large, i_out_large, i_gt_sliced, i_out_sliced):
        if ENABLED['adversarial']:
            def d(xa, xb):
                return torch.sigmoid(xa - xb.mean())

            def l(wgan: Discriminator, xr, xf):
                assert not any(p.requires_grad for p in wgan.parameters())
                yr = wgan(xr)
                yf = wgan(xf)
                return -torch.log(1 - d(yr, yf) + 0.0001).mean() - torch.log(d(yf, yr) + 0.0001).mean()

            loss = l(self.wgan_global, i_gt_large, i_out_large)
            if self.wgan_local is not None:
                loss += l(self.wgan_local, i_gt_sliced, i_out_sliced)
            else:
                # Compensate for missing local loss
                loss *= 2

            assert not torch.isnan(loss)
            return loss


class StylePerceptualLoss(nn.Module):
    """
    Combination of the style and perceptual loss functions.
    Since these use the same activation maps, it is more efficient to compute them together.

    Perceptual loss:
    https://github.com/ceshine/fast-neural-style --> loss_network.py
    """

    def __init__(self):
        super().__init__()
        self.vgg_layers = vgg.vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '6': "relu2_1",
            '11': "relu3_1",
            '18': "relu4_1",
            '25': "relu5_1",
        }
        self.last_vgg_layer = max(int(k) for k in self.layer_name_mapping.keys())

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

        return gram / (d * h * w)

    def forward(self, i_gt, i_out):
        """
        Calculate perceptual loss.

        :param i_gt: 3x256x256 tensor representing the unmasked source image
        :param i_out: 3x256x256 tensor, the output of the network on the masked image
        :return: A float
        """
        activation_gt = {}
        activation_out = {}

        if ENABLED['style'] or ENABLED['perceptual']:
            for name, module in self.vgg_layers._modules.items():
                i_gt = module(i_gt)
                i_out = module(i_out)
                if name in self.layer_name_mapping:
                    activation_gt[self.layer_name_mapping[name]] = i_gt
                    activation_out[self.layer_name_mapping[name]] = i_out
                if int(name) > self.last_vgg_layer:
                    break

        percept_loss = 0
        if ENABLED['perceptual']:
            percept_loss += self.percept_loss1(activation_gt['relu1_1'], activation_out['relu1_1'])
            percept_loss += self.percept_loss2(activation_gt['relu2_1'], activation_out['relu2_1'])
            percept_loss += self.percept_loss3(activation_gt['relu3_1'], activation_out['relu3_1'])
            percept_loss += self.percept_loss4(activation_gt['relu4_1'], activation_out['relu4_1'])
            percept_loss += self.percept_loss5(activation_gt['relu5_1'], activation_out['relu5_1'])
            percept_loss /= 5

        style_loss = 0
        if ENABLED['style']:
            gram_gt = {l: self.gram_matrix(x) for l, x in activation_gt.items()}
            gram_out = {l: self.gram_matrix(x) for l, x in activation_out.items()}
            style_loss += self.style_loss1(gram_gt['relu1_1'], gram_out['relu1_1'])
            style_loss += self.style_loss2(gram_gt['relu2_1'], gram_out['relu2_1'])
            style_loss += self.style_loss3(gram_gt['relu3_1'], gram_out['relu3_1'])
            style_loss += self.style_loss4(gram_gt['relu4_1'], gram_out['relu4_1'])
            style_loss += self.style_loss5(gram_gt['relu5_1'], gram_out['relu5_1'])
            style_loss /= 5

        return style_loss, percept_loss


