import torch
from torch import nn

from partial_conv import PartialConv2d


class Branch(nn.Module):
    """
    Description from paper:
    We design two branches (i.e., the structure branch and the texture branch)
    to separately perform hole filling on Fte and Fst. The architectures of these
    two branches are the same. In each branch, there are 3 parallel streams to fill
    holes in multiple scales. Each stream consists of 5 partial convolutions [20]
    with the same kernel size while the kernel size differs among different streams.
    """

    def __init__(self):
        super().__init__()

        self.stream7_1 = PartialConv2d(kernel_size=(7, 7))
        self.stream7_2 = PartialConv2d(kernel_size=(7, 7))
        self.stream7_3 = PartialConv2d(kernel_size=(7, 7))
        self.stream7_4 = PartialConv2d(kernel_size=(7, 7))
        self.stream7_5 = PartialConv2d(kernel_size=(7, 7))

        self.stream5_1 = PartialConv2d(kernel_size=(5, 5))
        self.stream5_2 = PartialConv2d(kernel_size=(5, 5))
        self.stream5_3 = PartialConv2d(kernel_size=(5, 5))
        self.stream5_4 = PartialConv2d(kernel_size=(5, 5))
        self.stream5_5 = PartialConv2d(kernel_size=(5, 5))

        self.stream3_1 = PartialConv2d(kernel_size=(3, 3))
        self.stream3_2 = PartialConv2d(kernel_size=(3, 3))
        self.stream3_3 = PartialConv2d(kernel_size=(3, 3))
        self.stream3_4 = PartialConv2d(kernel_size=(3, 3))
        self.stream3_5 = PartialConv2d(kernel_size=(3, 3))

        self.combining_conv = nn.Conv2d(kernel_size=(1,1))

    def forward(self, stream_7, stream_5, stream_3):
        stream_7 = self.stream7_1(stream_7)
        stream_7 = self.stream7_2(stream_7)
        stream_7 = self.stream7_3(stream_7)
        stream_7 = self.stream7_4(stream_7)
        stream_7 = self.stream7_5(stream_7)

        stream_5 = self.stream5_1(stream_5)
        stream_5 = self.stream5_2(stream_5)
        stream_5 = self.stream5_3(stream_5)
        stream_5 = self.stream5_4(stream_5)
        stream_5 = self.stream5_5(stream_5)

        stream_3 = self.stream3_1(stream_3)
        stream_3 = self.stream3_2(stream_3)
        stream_3 = self.stream3_3(stream_3)
        stream_3 = self.stream3_4(stream_3)
        stream_3 = self.stream3_5(stream_3)

        concat = torch.cat((stream_3, stream_5, stream_7))
        return self.combining_conv(concat)
