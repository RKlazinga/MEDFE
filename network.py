from torch import nn
import torch
from torch.utils import tensorboard, data
from PIL import Image
import numpy as np

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

        # TODO res block
        self.res_block = ResBlock(512, 512, kernel_size=(4, 4), dilation=(2, 2))

        self.deconv5 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(1024, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(512, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(256, 64, (4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(128, 4, (4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x1 = self.conv1.forward(x)
        x2 = self.conv2.forward(x1)
        x3 = self.conv3.forward(x2)
        x4 = self.conv4.forward(x3)
        x5 = self.conv5.forward(x4)
        x6 = self.conv6.forward(x5)

        #TODO apply res block to x6
        # - output should have same shape as x5
        x_res = self.res_block.forward(x6)

        x5c = torch.cat((x_res, x5), dim=1)
        y5 = self.deconv5.forward(x5c)
        y4 = self.deconv4.forward(torch.cat((y5, x4), dim=1))
        y3 = self.deconv3.forward(torch.cat((y4, x3), dim=1))
        y2 = self.deconv2.forward(torch.cat((y3, x2), dim=1))
        y1 = self.deconv1.forward(torch.cat((y2, x1), dim=1))
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
