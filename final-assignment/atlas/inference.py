"""Small helpers shared by the predictors."""
import numpy as np
import torch
import torch.nn.functional as F


def frame_to_tensor(frame: np.ndarray, size: tuple[int, int], device) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (1, 3, h, w) float in [0, 1] on `device`, resized to `size`."""
    x = torch.from_numpy(frame).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def logits_to_mask(logits: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    """(1, C, h, w) logits -> (H, W) uint8 train-ID mask at the original frame size."""
    logits = F.interpolate(logits.float(), size=out_size, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()


def load_state_dict(model: torch.nn.Module, path: str, device) -> torch.nn.Module:
    """Load weights saved by the training scripts and put the model in eval mode."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()
