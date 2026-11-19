"""Submission interface for the consistency track.

The server calls `reset()` at the start of every clip and then `step(frame)` once per frame,
in temporal order. Only past frames may be used, so keep any memory you need in `self`.
"""
import numpy as np
import torch

from atlas.data import SIZE
from atlas.inference import frame_to_tensor, load_state_dict, logits_to_mask
from consistency.model import Model

INPUT_SIZE = SIZE  # resolution the model was trained on (H, W); change it if you train at another size


class Predictor:
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = load_state_dict(Model(), checkpoint, device)
        self.state = None

    def reset(self):
        """Forget the previous clip."""
        self.state = None

    @torch.no_grad()
    def step(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB frame -> (H, W) uint8 mask with train IDs, same H and W."""
        x = frame_to_tensor(frame, INPUT_SIZE, self.device)
        logits, self.state = self.model(x, self.state)
        return logits_to_mask(logits, frame.shape[:2])
