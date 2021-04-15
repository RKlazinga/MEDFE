from torch import nn
import torch


from network.branch import Branch
from network.conv_unit import ConvUnit, DeConvUnit
from network.res_block import ResBlock
from network.bpa import Bpa


class MEDFE(nn.Module):
    """
    Complete network topology as described by the paper.
    """
    def __init__(self, batch_norm=True, use_bpa=False, use_branch=True, use_res=True, branch_channels=512, channels=64):
        """
        Initialise the network.

        :param batch_norm: Whether to use batch normalisation (unspecified by paper)
        :param use_bpa: Whether to use or skip the BPA module
        :param use_branch: Whether to use the feature and texture branches
        :param use_res: Whether to use the Res blocks situated between the encoder and decoder
        :param branch_channels: Number of channels used in tensors in the branches (unspecified by paper)
        :param channels: Multiplier for number of channels used in encoder/decoder (unspecified by paper)
        """
        super().__init__()

        self.batch_norm = batch_norm
        self.use_bpa = use_bpa
        self.use_branch = use_branch
        self.use_res = use_res

        # ENCODER
        self.conv1 = ConvUnit(4, channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = ConvUnit(channels, 2 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = ConvUnit(2 * channels, 4 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = ConvUnit(4 * channels, 8 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv5 = ConvUnit(8 * channels, 8 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv6 = ConvUnit(8 * channels, 8 * channels, (4, 4), stride=(2, 2), padding=(1, 1))

        self.res_block1 = ResBlock(8 * channels, 8 * channels, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block2 = ResBlock(8 * channels, 8 * channels, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block3 = ResBlock(8 * channels, 8 * channels, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block4 = ResBlock(8 * channels, 8 * channels, kernel_size=(2, 2), dilation=(2, 2))

        # TEXTURE BRANCH
        self.texture_branch = Branch(branch_channels)

        # downscaling convolutions to provide features for texture branch
        self.tex_branch_downscale_1 = ConvUnit(channels, branch_channels, kernel_size=(4, 4), stride=(4, 4), use_batch_norm=False)
        self.tex_branch_downscale_2 = ConvUnit(2 * channels, branch_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), use_batch_norm=False)
        self.tex_branch_downscale_3 = ConvUnit(4 * channels, branch_channels, kernel_size=(1, 1), use_batch_norm=False)
        # 1x1 convolution to combine the features above
        self.tex_branch_combine = ConvUnit(3*branch_channels, branch_channels, kernel_size=(1, 1), use_batch_norm=False)

        # 1x1 convolution to produce intermediate image
        self.tex_branch_to_img = ConvUnit(branch_channels, 3, kernel_size=(1, 1), use_batch_norm=False)
        self.tex_branch_img = None

        # STRUCTURE BRANCH
        self.structure_branch = Branch(branch_channels)

        # upscaling convolutions to provide features for structure branch
        self.struct_branch_upscale_4 = DeConvUnit(8 * channels, branch_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), use_batch_norm=False)
        self.struct_branch_upscale_5 = DeConvUnit(8 * channels, branch_channels, kernel_size=(4, 4), stride=(4, 4), use_batch_norm=False)
        self.struct_branch_upscale_6 = DeConvUnit(8 * channels, branch_channels, kernel_size=(8, 8), stride=(8, 8), use_batch_norm=False)
        # 1x1 convolution to combine the features above
        self.struct_branch_combine = ConvUnit(3 * branch_channels, branch_channels, kernel_size=(1, 1), use_batch_norm=False)

        # 1x1 convolution to produce intermediate image
        self.struct_branch_to_img = ConvUnit(branch_channels, 3, kernel_size=(1,1), use_batch_norm=False)
        self.struct_branch_img = None

        # 1x1 convolution to combine branch outputs
        self.branch_combiner = ConvUnit(2 * branch_channels, branch_channels, kernel_size=(1, 1))

        # spatial and channel equalisation operator
        self.bpa = Bpa((1, branch_channels, 32, 32))

        # down/upscale operators to resize branch output to appropriate dimensions for the decoder
        self.branch_scale_6 = ConvUnit(branch_channels, 8 * channels, kernel_size=(8, 8), stride=(8, 8), use_batch_norm=False)
        self.branch_scale_6_bn = nn.BatchNorm2d(8 * channels)

        self.branch_scale_5 = ConvUnit(branch_channels, 8 * channels, kernel_size=(4, 4), stride=(4, 4), use_batch_norm=False)
        self.branch_scale_5_bn = nn.BatchNorm2d(8 * channels)

        self.branch_scale_4 = ConvUnit(branch_channels, 8 * channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), use_batch_norm=False)
        self.branch_scale_4_bn = nn.BatchNorm2d(8 * channels)

        self.branch_scale_3 = ConvUnit(branch_channels, 4 * channels, kernel_size=(1, 1), use_batch_norm=False)
        self.branch_scale_3_bn = nn.BatchNorm2d(4 * channels)

        self.branch_scale_2 = DeConvUnit(branch_channels, 2 * channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), use_batch_norm=False)
        self.branch_scale_2_bn = nn.BatchNorm2d(2 * channels)

        self.branch_scale_1 = DeConvUnit(branch_channels, channels, kernel_size=(4, 4), stride=(4, 4), use_batch_norm=False)
        self.branch_scale_1_bn = nn.BatchNorm2d(channels)

        # DECODER
        self.deconv6 = DeConvUnit(8 * channels, 8 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv5 = DeConvUnit(16 * channels, 8 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv4 = DeConvUnit(16 * channels, 4 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv3 = DeConvUnit(8 * channels, 2 * channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv2 = DeConvUnit(4 * channels, channels, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv1 = DeConvUnit(2 * channels, 3, (4, 4), stride=(2, 2), padding=(1, 1), use_batch_norm=False)

        self.mask = None

    def set_mask(self, mask):
        old_mask = self.mask
        self.mask = mask

        self.texture_branch.set_mask(mask)
        self.structure_branch.set_mask(mask)

        return old_mask

    def forward(self, x):
        # apply encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # apply res blocks
        if self.use_res:
            x_res1 = self.res_block1(x6)
            x_res2 = self.res_block2(x_res1)
            x_res3 = self.res_block3(x_res2)
            x_res4 = self.res_block4(x_res3)
        else:
            x_res4 = x6.clone()

        if self.use_branch:
            # TEXTURE BRANCH
            # scale x1, x2 and x3 to 32x32
            tex_branch_input = torch.cat((
                self.tex_branch_downscale_1(x1),
                self.tex_branch_downscale_2(x2),
                self.tex_branch_downscale_3(x3)
            ), dim=1)
            tex_branch_input = self.tex_branch_combine(tex_branch_input)
            f_fte = self.texture_branch(tex_branch_input)
            self.tex_branch_img = self.tex_branch_to_img(f_fte)

            # STRUCTURE BRANCH
            # upscale x4, x5 and x6 to 32x32
            struct_branch_input = torch.cat((
                self.struct_branch_upscale_4(x4),
                self.struct_branch_upscale_5(x5),
                self.struct_branch_upscale_6(x6)
            ), dim=1)
            struct_branch_input = self.struct_branch_combine(struct_branch_input)
            f_fst = self.structure_branch(struct_branch_input)
            self.struct_branch_img = self.struct_branch_to_img(f_fst)

            # concatenate and combine branches
            f_sf = self.branch_combiner(torch.cat((f_fst, f_fte), dim=1))

            if self.use_bpa:
                f_sf = self.bpa(f_sf)
            # mask branch output (note that mask is inverted)
            # this is so that unstable edge-of-mask outputs are not passed on to the decoder
            # these cause some artifacts on the edges of the mask
            f_sf = f_sf * self.structure_branch.mask

            # compute skip connections
            x_res = self.branch_scale_6_bn(x_res4 + self.branch_scale_6(f_sf))
            x5_skip = self.branch_scale_5_bn(x5 + self.branch_scale_5(f_sf))
            x4_skip = self.branch_scale_4_bn(x4 + self.branch_scale_4(f_sf))
            x3_skip = self.branch_scale_3_bn(x3 + self.branch_scale_3(f_sf))
            x2_skip = self.branch_scale_2_bn(x2 + self.branch_scale_2(f_sf))
            x1_skip = self.branch_scale_1_bn(x1 + self.branch_scale_1(f_sf))
        else:
            x_res = x_res4
            x5_skip = x5.clone()
            x4_skip = x4.clone()
            x3_skip = x3.clone()
            x2_skip = x2.clone()
            x1_skip = x1.clone()

        # apply decoder
        y6 = self.deconv6(x_res)
        y5 = self.deconv5(torch.cat((y6, x5_skip), dim=1))
        y4 = self.deconv4(torch.cat((y5, x4_skip), dim=1))
        y3 = self.deconv3(torch.cat((y4, x3_skip), dim=1))
        y2 = self.deconv2(torch.cat((y3, x2_skip), dim=1))
        y1 = self.deconv1(torch.cat((y2, x1_skip), dim=1))

        return y1
