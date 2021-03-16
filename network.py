from torch import nn
import torch
from torch.utils import tensorboard

class MEDFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 64, (4, 4), stride=2)
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=2)
        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=2)
        self.conv5 = nn.Conv2d(512, 512, (4, 4), stride=2)
        self.conv6 = nn.Conv2d(512, 512, (4, 4), stride=2)

        # TODO res block

        self.deconv5 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=2)
        self.deconv4 = nn.ConvTranspose2d(1024, 256, (4, 4), stride=2)
        self.deconv3 = nn.ConvTranspose2d(512, 128, (4, 4), stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 64, (4, 4), stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 4, (4, 4), stride=2)

    def forward(self, x):
        x1 = self.conv1.forward(x)
        x2 = self.conv1.forward(x1)
        x3 = self.conv1.forward(x2)
        x4 = self.conv1.forward(x3)
        x5 = self.conv1.forward(x4)
        x6 = self.conv1.forward(x5)

        #TODO apply res block to x6
        # - output should have same shape as x5
        x_res = torch.tensor(x6)

        y5 = self.deconv5.forward(torch.cat((x_res, x5)))
        y4 = self.deconv4.forward(torch.cat((y5, x4)))
        y3 = self.deconv3.forward(torch.cat((y4, x3)))
        y2 = self.deconv2.forward(torch.cat((y3, x2)))
        y1 = self.deconv1.forward(torch.cat((y2, x1)))

medfe = MEDFE()
writer = tensorboard.SummaryWriter("tensorboard_logs")
writer.add_graph(medfe)
writer.close()
