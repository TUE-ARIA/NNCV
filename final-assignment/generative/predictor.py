"""Submission interface for the generative track.

The server calls `generate(masks)` once per clip of 16 consecutive masks (256x384, train IDs)
and expects 16 RGB frames back. You may use all masks of the clip at once.
"""
import numpy as np
import torch

from atlas.classes import NUM_CLASSES
from atlas.data import one_hot
from atlas.inference import load_state_dict
from generative.model import Generator


class Predictor:
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = load_state_dict(Generator(), checkpoint, device)

    @torch.no_grad()
    def generate(self, masks: np.ndarray) -> np.ndarray:
        """(T, H, W) uint8 masks -> (T, H, W, 3) uint8 RGB frames. The baseline treats every frame independently."""
        cond = one_hot(torch.from_numpy(masks).to(self.device), NUM_CLASSES)
        frames = self.model(cond)                                        # (T, 3, H, W) in [-1, 1]
        frames = ((frames + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()
