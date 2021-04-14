import torch
from torch import nn
import torch.nn.functional as F

from network.partial_conv import PartialConv2d


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

        self.stream7_1 = PartialConv2d(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.relu7_1 = nn.ReLU()
        self.batch_norm7_1 = nn.BatchNorm2d(self.input_size)

        self.stream7_2 = PartialConv2d(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.relu7_2 = nn.ReLU()
        self.batch_norm7_2 = nn.BatchNorm2d(self.input_size)

        self.stream7_3 = PartialConv2d(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.relu7_3 = nn.ReLU()
        self.batch_norm7_3 = nn.BatchNorm2d(self.input_size)

        self.stream7_4 = PartialConv2d(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.relu7_4 = nn.ReLU()
        self.batch_norm7_4 = nn.BatchNorm2d(self.input_size)

        self.stream7_5 = PartialConv2d(self.input_size, self.input_size, kernel_size=(7, 7), padding=(3, 3))
        self.relu7_5 = nn.ReLU()
        self.batch_norm7_5 = nn.BatchNorm2d(self.input_size)

        self.stream5_1 = PartialConv2d(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.relu5_1 = nn.ReLU()
        self.batch_norm5_1 = nn.BatchNorm2d(self.input_size)

        self.stream5_2 = PartialConv2d(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.relu5_2 = nn.ReLU()
        self.batch_norm5_2 = nn.BatchNorm2d(self.input_size)

        self.stream5_3 = PartialConv2d(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.relu5_3 = nn.ReLU()
        self.batch_norm5_3 = nn.BatchNorm2d(self.input_size)

        self.stream5_4 = PartialConv2d(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.relu5_4 = nn.ReLU()
        self.batch_norm5_4 = nn.BatchNorm2d(self.input_size)

        self.stream5_5 = PartialConv2d(self.input_size, self.input_size, kernel_size=(5, 5), padding=(2, 2))
        self.relu5_5 = nn.ReLU()
        self.batch_norm5_5 = nn.BatchNorm2d(self.input_size)

        self.stream3_1 = PartialConv2d(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.relu3_1 = nn.ReLU()
        self.batch_norm3_1 = nn.BatchNorm2d(self.input_size)

        self.stream3_2 = PartialConv2d(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.relu3_2 = nn.ReLU()
        self.batch_norm3_2 = nn.BatchNorm2d(self.input_size)

        self.stream3_3 = PartialConv2d(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.relu3_3 = nn.ReLU()
        self.batch_norm3_3 = nn.BatchNorm2d(self.input_size)

        self.stream3_4 = PartialConv2d(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.relu3_4 = nn.ReLU()
        self.batch_norm3_4 = nn.BatchNorm2d(self.input_size)

        self.stream3_5 = PartialConv2d(self.input_size, self.input_size, kernel_size=(3, 3), padding=(1, 1))
        self.relu3_5 = nn.ReLU()
        self.batch_norm3_5 = nn.BatchNorm2d(self.input_size)

        self.combining_conv = nn.Conv2d(3 * self.input_size, self.input_size, kernel_size=(1, 1))
        self.combine_relu = nn.ReLU()
        self.combine_batch_norm = nn.BatchNorm2d(self.input_size)

    def set_mask(self, mask):
        n = mask.shape[0]
        self.mask = 1 - F.interpolate(mask.reshape(n, 1, 256, 256).float(), size=(32, 32))

    def forward(self, f):
        stream_7 = self.stream7_1(f, mask_in=self.mask)
        stream_7 = self.batch_norm7_1(self.relu7_1(stream_7))

        stream_7 = self.stream7_2(stream_7, mask_in=self.mask)
        stream_7 = self.batch_norm7_2(self.relu7_2(stream_7))

        stream_7 = self.stream7_3(stream_7, mask_in=self.mask)
        stream_7 = self.batch_norm7_3(self.relu7_3(stream_7))

        stream_7 = self.stream7_4(stream_7, mask_in=self.mask)
        stream_7 = self.batch_norm7_4(self.relu7_4(stream_7))

        stream_7 = self.stream7_5(stream_7, mask_in=self.mask)
        stream_7 = self.batch_norm7_5(self.relu7_5(stream_7))

        stream_5 = self.stream5_1(f, mask_in=self.mask)
        stream_5 = self.batch_norm5_1(self.relu5_1(stream_5))

        stream_5 = self.stream5_2(stream_5, mask_in=self.mask)
        stream_5 = self.batch_norm5_2(self.relu5_2(stream_5))

        stream_5 = self.stream5_3(stream_5, mask_in=self.mask)
        stream_5 = self.batch_norm5_3(self.relu5_3(stream_5))

        stream_5 = self.stream5_4(stream_5, mask_in=self.mask)
        stream_5 = self.batch_norm5_4(self.relu5_4(stream_5))

        stream_5 = self.stream5_5(stream_5, mask_in=self.mask)
        stream_5 = self.batch_norm5_5(self.relu5_5(stream_5))

        stream_3 = self.stream3_1(f, mask_in=self.mask)
        stream_3 = self.batch_norm3_1(self.relu3_1(stream_3))

        stream_3 = self.stream3_2(stream_3, mask_in=self.mask)
        stream_3 = self.batch_norm3_2(self.relu3_2(stream_3))

        stream_3 = self.stream3_3(stream_3, mask_in=self.mask)
        stream_3 = self.batch_norm3_3(self.relu3_3(stream_3))

        stream_3 = self.stream3_4(stream_3, mask_in=self.mask)
        stream_3 = self.batch_norm3_4(self.relu3_4(stream_3))

        stream_3 = self.stream3_5(stream_3, mask_in=self.mask)
        stream_3 = self.batch_norm3_5(self.relu3_5(stream_3))

        concat = torch.cat((stream_3, stream_5, stream_7), dim=1)
        combined = self.combining_conv(concat)

        return combined
