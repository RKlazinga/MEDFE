from torch import nn
import torch


from branch import Branch
from res_block import ResBlock
from bpa import Bpa


class MEDFE(nn.Module):
    def __init__(self, batch_norm=True, use_bpa=False, use_branch=True, use_res=True, branch_channels=10):
        super().__init__()

        self.batch_norm = batch_norm
        self.use_bpa = use_bpa
        self.use_branch = use_branch
        self.use_res = use_res

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(64)
            self.batch_norm2 = nn.BatchNorm2d(128)
            self.batch_norm3 = nn.BatchNorm2d(256)
            self.batch_norm4 = nn.BatchNorm2d(512)
            self.batch_norm5 = nn.BatchNorm2d(512)
            self.batch_norm6 = nn.BatchNorm2d(512)

            self.batch_norm6_de = nn.BatchNorm2d(512)
            self.batch_norm5_de = nn.BatchNorm2d(512)
            self.batch_norm4_de = nn.BatchNorm2d(256)
            self.batch_norm3_de = nn.BatchNorm2d(128)
            self.batch_norm2_de = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(4, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu6 = nn.ReLU()

        self.res_block1 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block2 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block3 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block4 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))

        self.tex_branch_downscale_1 = nn.Conv2d(64, branch_channels, kernel_size=(4, 4), stride=(4, 4))
        self.tex_branch_downscale_2 = nn.Conv2d(128, branch_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tex_branch_downscale_3 = nn.Conv2d(256, branch_channels, kernel_size=(1, 1))
        self.tex_branch_combine = nn.Conv2d(3*branch_channels, branch_channels, kernel_size=(1, 1))
        self.texture_branch = Branch(branch_channels)
        self.tex_branch_to_img = nn.Conv2d(branch_channels, 3, kernel_size=(1, 1))
        self.tex_branch_img = None

        self.struct_branch_upscale_4 = nn.ConvTranspose2d(512, branch_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.struct_branch_upscale_5 = nn.ConvTranspose2d(512, branch_channels, kernel_size=(4, 4), stride=(4, 4))
        self.struct_branch_upscale_6 = nn.ConvTranspose2d(512, branch_channels, kernel_size=(8, 8), stride=(8, 8))
        self.struct_branch_combine = nn.Conv2d(3 * branch_channels, branch_channels, kernel_size=(1, 1))
        self.structure_branch = Branch(branch_channels)
        self.struct_branch_to_img = nn.Conv2d(branch_channels, 3, kernel_size=(1,1))
        self.struct_branch_img = None

        self.branch_combiner = nn.Conv2d(2 * branch_channels, branch_channels, kernel_size=(1, 1))

        self.bpa = Bpa((1, branch_channels, 32, 32))

        self.branch_scale_6 = nn.Conv2d(branch_channels, 512, kernel_size=(8, 8), stride=(8, 8))
        self.branch_scale_5 = nn.Conv2d(branch_channels, 512, kernel_size=(4, 4), stride=(4, 4))
        self.branch_scale_4 = nn.Conv2d(branch_channels, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_3 = nn.Conv2d(branch_channels, 256, kernel_size=(1, 1))
        self.branch_scale_2 = nn.ConvTranspose2d(branch_channels, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_1 = nn.ConvTranspose2d(branch_channels, 64, kernel_size=(4, 4), stride=(4, 4))

        self.deconv6 = nn.ConvTranspose2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu_de_6 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu_de_5 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(1024, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu_de_4 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(512, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu_de_3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu_de_2 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(128, 3, (4, 4), stride=(2, 2), padding=(1, 1))
        self.final_activation = nn.ReLU()

        self.mask = None

    def set_mask(self, mask):
        old_mask = self.mask
        self.mask = mask

        self.texture_branch.set_mask(mask)
        self.structure_branch.set_mask(mask)

        return old_mask

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        if self.batch_norm:
            self.batch_norm1(x1)
        x2 = self.relu2(self.conv2(x1))
        if self.batch_norm:
            self.batch_norm2(x2)
        x3 = self.relu3(self.conv3(x2))
        if self.batch_norm:
            self.batch_norm3(x3)
        x4 = self.relu4(self.conv4(x3))
        if self.batch_norm:
            self.batch_norm4(x4)
        x5 = self.relu5(self.conv5(x4))
        if self.batch_norm:
            self.batch_norm5(x5)
        x6 = self.relu6(self.conv6(x5))
        if self.batch_norm:
            self.batch_norm6(x6)

        if self.use_res:
            x_res1 = self.res_block1(x6)
            x_res2 = self.res_block2(x_res1)
            x_res3 = self.res_block3(x_res2)
            x_res4 = self.res_block4(x_res3)
        else:
            x_res4 = x6.clone()

        if self.use_branch:
            # texture branch: scale x1, x2 and x3 to 32x32
            tex_branch_input = torch.cat((
                self.tex_branch_downscale_1(x1),
                self.tex_branch_downscale_2(x2),
                self.tex_branch_downscale_3(x3)
            ), dim=1)
            tex_branch_input = self.tex_branch_combine(tex_branch_input)
            f_fte = self.texture_branch(tex_branch_input)
            self.tex_branch_img = self.tex_branch_to_img(f_fte)

            # structure branch: upscale x4, x5 and x6 to 32x32
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

            x_res = x_res4 + self.branch_scale_6(f_sf)
            x5_skip = x5 + self.branch_scale_5(f_sf)
            x4_skip = x4 + self.branch_scale_4(f_sf)
            x3_skip = x3 + self.branch_scale_3(f_sf)
            x2_skip = x2 + self.branch_scale_2(f_sf)
            x1_skip = x1 + self.branch_scale_1(f_sf)
        else:
            x_res = x_res4
            x5_skip = x5.clone()
            x4_skip = x4.clone()
            x3_skip = x3.clone()
            x2_skip = x2.clone()
            x1_skip = x1.clone()

        y6 = self.relu_de_6(self.deconv6(x_res))
        if self.batch_norm:
            y6 = self.batch_norm6_de(y6)
        y5 = self.relu_de_5(self.deconv5(torch.cat((y6, x5_skip), dim=1)))
        if self.batch_norm:
            y5 = self.batch_norm5_de(y5)
        y4 = self.relu_de_4(self.deconv4(torch.cat((y5, x4_skip), dim=1)))
        if self.batch_norm:
            y4 = self.batch_norm4_de(y4)
        y3 = self.relu_de_3(self.deconv3(torch.cat((y4, x3_skip), dim=1)))
        if self.batch_norm:
            y3 = self.batch_norm3_de(y3)
        y2 = self.relu_de_2(self.deconv2(torch.cat((y3, x2_skip), dim=1)))
        if self.batch_norm:
            y2 = self.batch_norm2_de(y2)
        y1 = self.deconv1(torch.cat((y2, x1_skip), dim=1))

        # y3 = self.deconv3(torch.cat((x3, x3), dim=1))
        # y2 = self.deconv2(torch.cat((self.relu_de_3(y3), x2), dim=1))
        # y1 = self.deconv1(torch.cat((self.relu_de_2(y2), x1), dim=1))

        return self.final_activation(y1)
