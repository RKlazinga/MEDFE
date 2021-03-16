from torch import nn


class Branch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        """
        After CNN feature reorganization, we design two branches (i.e., the structure
        branch and the texture branch) to separately perform hole filling on Fte and Fst.
        The architectures of these two branches are the same. In each branch, there are 3
        parallel streams to fill holes in multiple scales. Each stream consists of 5 partial
        convolutions [20] with the same kernel size while the kernel size differs among
        different streams.
        """