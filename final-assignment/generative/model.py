"""Baseline for the generative track: pix2pix (Isola et al., 2017), applied frame by frame.

Generator: U-Net that maps a one-hot mask (B, num_classes, 256, 384) to an RGB frame (B, 3, 256, 384) in [-1, 1].
Discriminator: PatchGAN that scores overlapping 70x70 patches of (mask, frame) pairs as real or fake.
"""
import torch
import torch.nn as nn

from atlas.classes import NUM_CLASSES


def down(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_ch, out_ch, dropout=False):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """U-Net generator with `depth` down- and up-sampling steps.

    Height and width must be divisible by 2**depth: 256x384 with depth 7 gives a 2x3 bottleneck.
    """

    def __init__(self, num_classes=NUM_CLASSES, base=64, depth=7):
        super().__init__()
        chs = [base * min(2 ** i, 8) for i in range(depth)]  # 64, 128, 256, 512, 512, ...
        self.downs = nn.ModuleList([down(num_classes, chs[0], norm=False)] +
                                   [down(chs[i - 1], chs[i], norm=i < depth - 1) for i in range(1, depth)])
        # each up block gets the previous output concatenated with the matching skip connection
        self.ups = nn.ModuleList([up(chs[-1], chs[-2], dropout=True)] +
                                 [up(chs[-1 - i] * 2, chs[-2 - i], dropout=i < 3) for i in range(1, depth - 1)])
        self.out = nn.Sequential(nn.ConvTranspose2d(chs[0] * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, mask_onehot):
        skips, x = [], mask_onehot
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = self.ups[0](skips[-1])
        for u, skip in zip(self.ups[1:], reversed(skips[1:-1])):
            x = u(torch.cat([x, skip], 1))
        return self.out(torch.cat([x, skips[0]], 1))


class Discriminator(nn.Module):
    """PatchGAN discriminator. Input: mask one-hot and RGB frame; output: a map of real/fake logits."""

    def __init__(self, num_classes=NUM_CLASSES, base=64):
        super().__init__()
        self.net = nn.Sequential(
            down(num_classes + 3, base, norm=False),
            down(base, base * 2),
            down(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False), nn.InstanceNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, mask_onehot, image):
        return self.net(torch.cat([mask_onehot, image], 1))
