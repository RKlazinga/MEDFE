import torch.nn as nn
import traceback


class Discriminator(nn.Module):
    def __init__(self, shape, name: str = None):
        super().__init__()

        self.name = name

        s, c, h, w = shape
        assert h == w, "The discriminator only supports square inputs"

        layers = []
        while h > 4:
            # From https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
            # > Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient
            # > penalty) is no longer valid in this setting, since we penalize the norm of the critic's gradient with
            # > respect to each input independently and not the entire batch. There is not good & fast implementation
            # > of layer normalization --> using per instance normalization nn.InstanceNorm2d()

            layers.append(nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(c * 2, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            assert (h // 2) * 2 == h
            h //= 2
            c *= 2

        layers.append(nn.Conv2d(in_channels=c, out_channels=1, kernel_size=4, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def with_grad(self):
        class GradCm:
            def __init__(self, module: nn.Module):
                self.module = module

            def __enter__(self):
                for p in self.module.parameters():
                    p.requires_grad = True

            def __exit__(self, exc_type, exc_val, exc_tb):
                for p in self.module.parameters():
                    p.requires_grad = False

        return GradCm(self)
