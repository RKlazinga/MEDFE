import torch
from torch import nn
import torch.nn.functional as F

from network.conv_unit import ConvUnit, PartialConvUnit


class Branch(nn.Module):
    """
    Description from paper:
    We design two branches (i.e., the structure branch and the texture branch)
    to separately perform hole filling on Fte and Fst. The architectures of these
    two branches are the same. In each branch, there are 3 parallel streams to fill
    holes in multiple scales. Each stream consists of 5 partial convolutions [20]
    with the same kernel size while the kernel size differs among different streams.
    """

    def __init__(self, input_size):
        super().__init__()
        self.mask = None
        self.input_size = input_size

        self.stream7_1 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.stream7_2 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.stream7_3 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.stream7_4 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.stream7_5 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))

        self.stream5_1 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.stream5_2 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.stream5_3 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.stream5_4 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.stream5_5 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))

        self.stream3_1 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.stream3_2 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.stream3_3 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.stream3_4 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.stream3_5 = PartialConvUnit(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))

        self.combining_conv = ConvUnit(3 * self.input_size, self.input_size, kernel_size=(1, 1))

    def set_mask(self, mask):
        n = mask.shape[0]
        self.mask = F.max_pool2d(1 - mask.reshape(n, 1, 256, 256).float(), (8, 8))

    def forward(self, f):
        stream_7 = self.stream7_1(f, mask_in=self.mask)
        stream_7 = self.stream7_2(stream_7, mask_in=self.mask)
        stream_7 = self.stream7_3(stream_7, mask_in=self.mask)
        stream_7 = self.stream7_4(stream_7, mask_in=self.mask)
        stream_7 = self.stream7_5(stream_7, mask_in=self.mask)

        stream_5 = self.stream5_1(f, mask_in=self.mask)
        stream_5 = self.stream5_2(stream_5, mask_in=self.mask)
        stream_5 = self.stream5_3(stream_5, mask_in=self.mask)
        stream_5 = self.stream5_4(stream_5, mask_in=self.mask)
        stream_5 = self.stream5_5(stream_5, mask_in=self.mask)

        stream_3 = self.stream3_1(f, mask_in=self.mask)
        stream_3 = self.stream3_2(stream_3, mask_in=self.mask)
        stream_3 = self.stream3_3(stream_3, mask_in=self.mask)
        stream_3 = self.stream3_4(stream_3, mask_in=self.mask)
        stream_3 = self.stream3_5(stream_3, mask_in=self.mask)

        concat = torch.cat((stream_3, stream_5, stream_7), dim=1)
        return self.combining_conv(concat)
