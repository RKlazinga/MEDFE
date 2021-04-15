from torch import nn
import torch


class ResBlock(nn.Module):
    """
    Residual block. Preserves size of input tensor.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv2d(self.in_size, self.out_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_size, self.out_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=(1, 1))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_temp = self.conv1(x)
        x_temp = self.relu1(x_temp)
        x_temp = self.conv2(x_temp)
        x = torch.add(x_temp, x)
        x = self.relu2(x)
        return x
