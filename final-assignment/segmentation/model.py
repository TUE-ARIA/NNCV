"""Baseline for the Dice track: a U-Net with an ImageNet-pretrained ResNet-18 encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from atlas.classes import NUM_CLASSES


class ConvBlock(nn.Sequential):
    """(3x3 conv -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    """Upsample to the skip connection's size, concatenate, then convolve."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Model(nn.Module):
    """U-Net with a ResNet-18 encoder.

    Input: (B, 3, H, W) RGB in [0, 1] (normalised inside the model). Multiples of 32 work best.
    Output: (B, num_classes, H, W) logits.

    Args:
        pretrained: load ImageNet weights for the encoder. Use True for training only; the
            predictor builds the model with the default (False) and loads your checkpoint.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)               # 1/2,  64 ch
        self.layers = nn.ModuleList([
            nn.Sequential(r.maxpool, r.layer1),                          # 1/4,  64 ch
            r.layer2,                                                    # 1/8,  128 ch
            r.layer3,                                                    # 1/16, 256 ch
            r.layer4,                                                    # 1/32, 512 ch
        ])
        self.ups = nn.ModuleList([Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 64)])
        self.head = nn.Conv2d(64, num_classes, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, x):
        """Return the encoder features from high to low resolution (the last one is the bottleneck)."""
        feats = [self.stem((x - self.mean) / self.std)]
        for layer in self.layers:
            feats.append(layer(feats[-1]))
        return feats

    def decode(self, feats, out_size):
        """Decode encoder features to logits of spatial size `out_size`."""
        x = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            x = up(x, skip)
        return F.interpolate(self.head(x), size=out_size, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.decode(self.encode(x), x.shape[-2:])
