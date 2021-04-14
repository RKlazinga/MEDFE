import torch
from torch import nn
import torch.nn.functional as F
import math


class RangeStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        n, c, h, w = x.data.shape
        unfolded_to_neigh = F.pad(x, (1, 1, 1, 1, 0, 0)).unfold(2, 3, 1).unfold(3, 3, 1)

        unfolded_left = unfolded_to_neigh.reshape(n, h * w, 3, 3, c)
        unfolded_mid = x.permute(0, 2, 3, 1).reshape(n, h * w, 1, 1, c)
        unfolded_right = unfolded_to_neigh.reshape(n, h*w, c, 3, 3)

        p1 = unfolded_left + unfolded_mid.repeat(1, 1, 3, 3, 1)
        p1_sum = torch.sum(p1, dim=4)
        p1_sum_softmaxed = self.softmax(p1_sum)

        final_grouped = p1_sum_softmaxed.reshape(n, h*w, 1, 3, 3).repeat(1, 1, c, 1, 1) + unfolded_right
        final = torch.sum(final_grouped, dim=(3, 4))

        return final.reshape(n, c, h, w)


class SpatialStep(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        n, c, w, h = input_shape
        assert w == h

        # With thanks to Adrian Sahlman:
        # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        x_coord = torch.arange(w+1)
        x_grid = x_coord.repeat(h+1).view(w+1, h+1)
        xy_grid = torch.stack([x_grid, x_grid.t()], dim=-1)

        mean = (w - 1) / 2
        var = (3 * w)**2  # Guess... Affects the "FOV" of the spatial step

        kernel = (1 / (2 * math.pi * var)) * torch.exp(-torch.sum((xy_grid - mean)**2, dim=-1) / (2 * var))
        norm_kernel = kernel / torch.sum(kernel)
        kernel_as_weights = norm_kernel.view(1, 1, w+1, h+1).repeat(c, 1, 1, 1)

        self.conv = nn.Conv2d(c, c, kernel_size=(33, 33), padding=w//2, padding_mode='reflect', bias=False, groups=c)
        self.conv.weight.data = kernel_as_weights
        self.conv.weight.requires_grad = False

    def forward(self, x):
        conved = self.conv(x)

        return conved


class Bpa(nn.Module):
    def __init__(self, input_shape):
        super(Bpa, self).__init__()
        self.range_step = RangeStep()
        self.spatial_step = SpatialStep(input_shape)

        self.combine_conv = nn.Conv2d(2 * input_shape[1], input_shape[1], (1, 1))

    def forward(self, x):
        ranged = self.range_step(x)
        spatialed = self.spatial_step(x)

        combined = torch.cat((ranged, spatialed), dim=1)

        return self.combine_conv(combined)
