"""Submission interface for the Dice track.

The server calls `reset()` at the start of every clip and then `step(frame)` once per frame,
in temporal order. Change the internals freely, but keep the class name and method signatures.
"""
import numpy as np
import torch

from atlas.data import SIZE
from atlas.inference import frame_to_tensor, load_state_dict, logits_to_mask
from segmentation.model import Model

INPUT_SIZE = SIZE  # resolution the model was trained on (H, W); change it if you train at another size


class Predictor:
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = load_state_dict(Model(), checkpoint, device)

    def reset(self):
        """Called before the first frame of each clip. This model keeps no state."""

    @torch.no_grad()
    def step(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB frame -> (H, W) uint8 mask with train IDs, same H and W."""
        x = frame_to_tensor(frame, INPUT_SIZE, self.device)
        return logits_to_mask(self.model(x), frame.shape[:2])
