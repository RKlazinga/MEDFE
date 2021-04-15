from torch import nn

from network.partial_conv import PartialConv2d


class ConvUnit(nn.Module):
    """
    Basic combination of convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, use_batch_norm=True, **kwargs):
        super().__init__()

        self.steps = [nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)]
        self.steps.append(nn.ReLU())
        if use_batch_norm:
            self.steps.append(nn.BatchNorm2d(out_channels))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class DeConvUnit(nn.Module):
    """
    Basic combination of DE-convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, use_batch_norm=True, **kwargs):
        super().__init__()

        self.steps = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, *args, **kwargs)]
        self.steps.append(nn.ReLU())
        if use_batch_norm:
            self.steps.append(nn.BatchNorm2d(out_channels))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class PartialConvUnit(nn.Module):
    """
    Basic combination of Partial Convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()

        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask_in=None):
        x = self.conv(x, mask_in)
        return self.bn(self.relu(x))
