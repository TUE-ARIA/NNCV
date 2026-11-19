"""Baseline for the consistency track: the Dice-track U-Net with a ConvGRU memory at the bottleneck.

The GRU carries information from previous frames to the current one, so the model is causal:
frame t only sees frames <= t.
"""
import torch
import torch.nn as nn

from atlas.classes import NUM_CLASSES
from segmentation.model import Model as UNet


class ConvGRUCell(nn.Module):
    """Convolutional GRU: a GRU where the matrix products are 3x3 convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)    # update (z) and reset (r) gates
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        h_new = torch.tanh(self.candidate(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * h_new


class Model(nn.Module):
    """U-Net + ConvGRU.

    `forward(x, state)` processes one frame: x is (B, 3, H, W) in [0, 1], state is the previous
    hidden state (None for the first frame). Returns (logits, new_state).
    `forward_clip(clip)` runs a whole (B, T, 3, H, W) clip and returns (B, T, C, H, W) logits.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.unet = UNet(num_classes, pretrained)
        self.gru = ConvGRUCell(512)

    def forward(self, x, state=None):
        feats = self.unet.encode(x)
        state = self.gru(feats[-1], state)
        feats[-1] = feats[-1] + state  # residual: the single-frame path still works without memory
        return self.unet.decode(feats, x.shape[-2:]), state

    def forward_clip(self, clip):
        state, outputs = None, []
        for t in range(clip.shape[1]):
            logits, state = self(clip[:, t], state)
            outputs.append(logits)
        return torch.stack(outputs, 1)
