"""Baseline for the real-time track: LR-ASPP with an ImageNet-pretrained MobileNetV3-Large backbone."""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from atlas.classes import NUM_CLASSES


class Model(nn.Module):
    """LR-ASPP MobileNetV3-Large (torchvision).

    Input: (B, 3, H, W) RGB in [0, 1] (normalised inside the model). Output: (B, num_classes, H, W) logits.

    Args:
        pretrained: load ImageNet weights for the backbone. Use True for training only.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)["out"]
