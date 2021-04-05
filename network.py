from torch import nn
import torch


from branch import Branch
from res_block import ResBlock
from bpa import Bpa


class MEDFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.relu1_cache = None

        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.relu2_cache = None

        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.relu3_cache = None

        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.relu4_cache = None

        self.conv5 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.relu5_cache = None

        self.conv6 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.relu6 = nn.ReLU()

        self.res_block1 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block2 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block3 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block4 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))

        self.tex_branch_downscale_1 = nn.Conv2d(64, 512, kernel_size=(4, 4), stride=(4, 4))
        self.tex_branch_downscale_2 = nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tex_branch_downscale_3 = nn.Conv2d(256, 512, kernel_size=(1, 1))
        self.tex_branch_combine = nn.Conv2d(3*512, 512, kernel_size=(1, 1))
        self.texture_branch = Branch(512)
        self.tex_branch_to_img = nn.Conv2d(512, 3, kernel_size=(1,1))
        self.tex_branch_img = None

        self.struct_branch_upscale_4 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.struct_branch_upscale_5 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(4, 4))
        self.struct_branch_upscale_6 = nn.ConvTranspose2d(512, 512, kernel_size=(8, 8), stride=(8, 8))
        self.struct_branch_combine = nn.Conv2d(3 * 512, 512, kernel_size=(1, 1))
        self.structure_branch = Branch(512)
        self.struct_branch_to_img = nn.Conv2d(512, 3, kernel_size=(1,1))
        self.struct_branch_img = None

        self.branch_combiner = nn.Conv2d(2 * 512, 512, kernel_size=(1, 1))

        self.bpa = Bpa((1, 512, 32, 32))

        self.branch_scale_6 = nn.Conv2d(512, 512, kernel_size=(8, 8), stride=(8, 8))
        self.branch_scale_5 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(4, 4))
        self.branch_scale_4 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_3 = nn.Conv2d(512, 256, kernel_size=(1, 1))
        self.branch_scale_2 = nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_1 = nn.ConvTranspose2d(512, 64, kernel_size=(4, 4), stride=(4, 4))

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
        self.relu_de_1 = nn.ReLU()

        self.mask = None

    def set_mask(self, mask):
        old_mask = self.mask
        self.mask = mask

        self.texture_branch.set_mask(mask)
        self.structure_branch.set_mask(mask)

        return old_mask

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        self.relu1_cache = x1.clone()
        x2 = self.relu2(self.conv2(x1))
        self.relu2_cache = x2.clone()
        x3 = self.relu3(self.conv3(x2))
        self.relu3_cache = x3.clone()
        x4 = self.relu4(self.conv4(x3))
        self.relu4_cache = x4.clone()
        x5 = self.relu5(self.conv5(x4))
        self.relu5_cache = x5.clone()
        x6 = self.relu6(self.conv6(x5))

        x_res1 = self.res_block1(x6)
        x_res2 = self.res_block2(x_res1)
        x_res3 = self.res_block3(x_res2)
        x_res4 = self.res_block4(x_res3)

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

        f_sf = self.bpa(f_sf)

        x_res = x_res4 + self.branch_scale_6(f_sf)
        x5_skip = x5 + self.branch_scale_5(f_sf)
        x4_skip = x4 + self.branch_scale_4(f_sf)
        x3_skip = x3 + self.branch_scale_3(f_sf)
        x2_skip = x2 + self.branch_scale_2(f_sf)
        x1_skip = x1 + self.branch_scale_1(f_sf)

        # TODO relu deconvolutions!
        y6 = self.deconv6(x_res)
        y5 = self.deconv5(torch.cat((self.relu_de_6(y6), x5_skip), dim=1))
        y4 = self.deconv4(torch.cat((self.relu_de_5(y5), x4_skip), dim=1))
        y3 = self.deconv3(torch.cat((self.relu_de_4(y4), x3_skip), dim=1))
        y2 = self.deconv2(torch.cat((self.relu_de_3(y3), x2_skip), dim=1))
        y1 = self.deconv1(torch.cat((self.relu_de_2(y2), x1_skip), dim=1))

        return self.relu_de_1(y1)


