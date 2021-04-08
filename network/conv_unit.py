from torch import nn


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, use_batch_norm=True, **kwargs):
        super().__init__()

        self.steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
            nn.ReLU()
        ]
        if use_batch_norm:
            self.steps.append(nn.BatchNorm2d(out_channels))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class DeConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, use_batch_norm=True, **kwargs):
        super().__init__()

        self.steps = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, *args, **kwargs),
            nn.ReLU()
        ]
        if use_batch_norm:
            self.steps.append(nn.BatchNorm2d(out_channels))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)
