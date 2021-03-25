from torch import nn
import torch
from torch.utils import tensorboard, data
from PIL import Image
import numpy as np

from branch import Branch
from res_block import ResBlock


class MEDFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv6 = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))

        self.res_block1 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block2 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block3 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))
        self.res_block4 = ResBlock(512, 512, kernel_size=(2, 2), dilation=(2, 2))

        self.tex_branch_downscale_1 = nn.Conv2d(64, 512, kernel_size=(4, 4), stride=(4, 4))
        self.tex_branch_downscale_2 = nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tex_branch_downscale_3 = nn.Conv2d(256, 512, kernel_size=(1, 1))
        self.tex_branch_combine = nn.Conv2d(3*512, 512, kernel_size=(1, 1))
        self.texture_branch = Branch(512)

        self.struct_branch_upscale_4 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.struct_branch_upscale_5 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(4, 4))
        self.struct_branch_upscale_6 = nn.ConvTranspose2d(512, 512, kernel_size=(8, 8), stride=(8, 8))
        self.struct_branch_combine = nn.Conv2d(3 * 512, 512, kernel_size=(1, 1))

        self.structure_branch = Branch(512)

        self.branch_combiner = nn.Conv2d(2 * 512, 512, kernel_size=(1, 1))

        self.branch_scale_6 = nn.Conv2d(512, 512, kernel_size=(8, 8), stride=(8, 8))
        self.branch_scale_5 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(4, 4))
        self.branch_scale_4 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_3 = nn.Conv2d(512, 256, kernel_size=(1, 1))
        self.branch_scale_2 = nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.branch_scale_1 = nn.ConvTranspose2d(512, 64, kernel_size=(4, 4), stride=(4, 4))

        self.deconv6 = nn.ConvTranspose2d(512, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv5 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(1024, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(512, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(256, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(128, 3, (4, 4), stride=(2, 2), padding=(1, 1))

    def set_mask(self, mask):
        self.texture_branch.set_mask(mask)
        self.structure_branch.set_mask(mask)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

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

        # structure branch: upscale x4, x5 and x6 to 32x32
        struct_branch_input = torch.cat((
            self.struct_branch_upscale_4(x4),
            self.struct_branch_upscale_5(x5),
            self.struct_branch_upscale_6(x6)
        ), dim=1)
        struct_branch_input = self.struct_branch_combine(struct_branch_input)
        f_fst = self.structure_branch(struct_branch_input)

        # concatenate and combine branches
        f_sf = self.branch_combiner(torch.cat((f_fst, f_fte), dim=1))

        # TODO apply channel equalization

        # TODO apply spatial equalization

        # TODO elementwise_add the outcome to all skip connections
        x_res = x_res4 + self.branch_scale_6(f_sf)
        x5_skip = x5 + self.branch_scale_5(f_sf)
        x4_skip = x4 + self.branch_scale_4(f_sf)
        x3_skip = x3 + self.branch_scale_3(f_sf)
        x2_skip = x2 + self.branch_scale_2(f_sf)
        x1_skip = x1 + self.branch_scale_1(f_sf)

        y6 = self.deconv6(x_res)
        y5 = self.deconv5(torch.cat((y6, x5_skip), dim=1))
        y4 = self.deconv4(torch.cat((y5, x4_skip), dim=1))
        y3 = self.deconv3(torch.cat((y4, x3_skip), dim=1))
        y2 = self.deconv2(torch.cat((y3, x2_skip), dim=1))
        y1 = self.deconv1(torch.cat((y2, x1_skip), dim=1))

        return y1


def main():
    mask = np.array(Image.open("30000.png").convert("L"))
    mask[mask <= 128] = 0
    mask[mask > 128] = 255

    im = np.array(Image.open("trial_image.png").convert("RGB"))
    im[mask == 0, :] = 0

    sample = torch.cat((torch.tensor(im), torch.tensor(mask.reshape(256, 256, 1))), dim=2)
    sample = torch.movedim(sample, 2, 0).float()

    train_loader = data.DataLoader([sample], batch_size=1, shuffle=True, num_workers=1)

    medfe = MEDFE()
    medfe.set_mask(torch.tensor(mask))
    writer = tensorboard.SummaryWriter("tensorboard_logs")

    writer.add_graph(medfe, next(iter(train_loader)))
    writer.close()


if __name__ == '__main__':
    main()
