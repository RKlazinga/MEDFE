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

        self.texture_branch = Branch()
        self.structure_branch = Branch()
        self.branch_combiner = nn.Conv2d(kernel_size=(1, 1))

        self.deconv5 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(1024, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(512, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(256, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(128, 4, (4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # TODO: apply res block to x6 instead of x5
        #  Because the output should have the same size as x5, just use x5 here until we can figure out what the paper
        #  is doing...
        x_res = x5
        x_res = self.res_block1(x_res)
        x_res = self.res_block2(x_res)
        x_res = self.res_block3(x_res)
        x_res = self.res_block4(x_res)

        f_fst = self.texture_branch(x1, x2, x3)
        f_fte = self.structure_branch(x4, x5, x6)

        # concatenate and combine branches
        f_sf = self.branch_combiner(torch.cat((f_fst, f_fte)))

        # TODO apply channel equalization

        # TODO apply spatial equalization

        # TODO elementwise_add the outcome to all skip connections

        y5 = self.deconv5(torch.cat((x_res, x5), dim=1))
        y4 = self.deconv4(torch.cat((y5, x4), dim=1))
        y3 = self.deconv3(torch.cat((y4, x3), dim=1))
        y2 = self.deconv2(torch.cat((y3, x2), dim=1))
        y1 = self.deconv1(torch.cat((y2, x1), dim=1))

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
    writer = tensorboard.SummaryWriter("tensorboard_logs")

    writer.add_graph(medfe, next(iter(train_loader)))
    writer.close()


if __name__ == '__main__':
    main()
