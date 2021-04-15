import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Uniform


class Discriminator(nn.Module):
    """
    Discriminator module for adversarial loss.
    """

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


def calc_gradient_penalty(wgan: Discriminator, real, fake, lambda_: float = 10):
    assert real.shape[1:] == fake.shape[1:]
    n, c, h, w = fake.shape

    eta = Uniform(0, 1).sample((n, 1, 1, 1)).expand(n, c, h, w)
    interpolated = eta * real[-n:] + (1 - eta) * fake
    interpolated.requires_grad = True

    prob_interpolated = wgan(interpolated)

    grad = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad = grad.view(grad.shape[0], -1)

    return ((grad.norm(2, dim=1) - 1) ** 2).mean() * lambda_


def train_discriminator(wgan: Discriminator, optimizer: optim.Optimizer, gt, out):
    assert not any(p.requires_grad for p in wgan.parameters())

    with wgan.with_grad():
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        optimizer.zero_grad()

        loss_real = wgan(gt.detach())
        loss_real = loss_real.mean()
        loss_real.backward(mone)

        loss_fake = wgan(out.detach())
        loss_fake = loss_fake.mean()
        loss_fake.backward(one)

        gradient_penalty = calc_gradient_penalty(wgan, real=gt.detach(), fake=out.detach())
        gradient_penalty.backward()

        optimizer.step()

    assert not any(p.requires_grad for p in wgan.parameters())

    loss = loss_fake - loss_real + gradient_penalty
    wasserstein = loss_real - loss_fake

    print(f"{wgan.name}: loss_real = {loss_real}; loss_fake = {loss_fake}; loss = {loss}; wasserstein = {wasserstein}")