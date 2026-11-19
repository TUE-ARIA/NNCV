"""Benchmark metrics. The challenge server uses exactly these functions.

Segmentation (all segmentation tracks):
    Dice: for every frame, the Dice per foreground class (train IDs 1-29) that appears in the
    ground truth or the prediction, averaged over those classes. Classes that appear in neither
    are skipped; a predicted class that is absent in the ground truth scores 0. Frames are
    averaged per clip, then clips are averaged.

Consistency:
    Change Dice: for each pair of neighbouring annotated frames (at most `max_gap` video frames
    apart) we mark the pixels whose label changes, in the prediction and in the ground truth.
    Change Dice is the Dice between these two change maps, pooled over all pairs of a clip.
    Flicker (changes the ground truth does not have) and lag (missing real changes) both lower it.
    Consistency score = (Dice + Change Dice) / 2.

Generative: see `frechet_distance`, `InceptionFeatures` and `VideoFeatures`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.classes import NUM_CLASSES


def frame_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> float | None:
    """Mean foreground Dice of one frame, or None if neither mask has foreground."""
    conf = np.bincount(gt.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel(),
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)
    inter = np.diag(conf)[1:]
    total = conf.sum(0)[1:] + conf.sum(1)[1:]   # |pred| + |gt| per class
    present = total > 0
    if not present.any():
        return None
    return float(np.mean(2 * inter[present] / total[present]))


class ClipScorer:
    """Accumulates the segmentation metrics for one clip. Call `add` for frames in temporal order."""

    def __init__(self, max_gap: int = 1):
        self.max_gap = max_gap
        self.dices = []
        self.change_inter = 0
        self.change_total = 0
        self.prev = None  # (frame index, pred, gt)

    def add(self, index: int, pred: np.ndarray, gt: np.ndarray):
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match ground truth {gt.shape}")
        d = frame_dice(pred, gt)
        if d is not None:
            self.dices.append(d)
        if self.prev is not None and index - self.prev[0] <= self.max_gap:
            change_pred, change_gt = pred != self.prev[1], gt != self.prev[2]
            self.change_inter += int((change_pred & change_gt).sum())
            self.change_total += int(change_pred.sum() + change_gt.sum())
        self.prev = (index, pred, gt)

    def dice(self) -> float | None:
        return float(np.mean(self.dices)) if self.dices else None

    def change_dice(self) -> float | None:
        """None if nothing changes in either the prediction or the ground truth."""
        return 2 * self.change_inter / self.change_total if self.change_total else None


def summarize(scorers: list[ClipScorer]) -> dict:
    """Average clip scores over all clips (clips without a defined value are skipped)."""
    dices = [d for s in scorers if (d := s.dice()) is not None]
    changes = [c for s in scorers if (c := s.change_dice()) is not None]
    dice = float(np.mean(dices)) if dices else 0.0
    change = float(np.mean(changes)) if changes else 0.0
    return {"dice": dice, "change_dice": change, "consistency_score": (dice + change) / 2, "num_clips": len(scorers)}


# ---------------------------------------------------------------------------
# Generative metrics
# ---------------------------------------------------------------------------

def frechet_distance(feats_a: torch.Tensor, feats_b: torch.Tensor) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets of shape (N, D)."""
    a, b = feats_a.double(), feats_b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = torch.cov(a.T), torch.cov(b.T)
    # trace(sqrt(cov_a @ cov_b)) = sum of square roots of the eigenvalues of cov_a @ cov_b
    eig = torch.linalg.eigvals(cov_a @ cov_b).real.clamp(min=0)
    dist = (mu_a - mu_b).square().sum() + cov_a.trace() + cov_b.trace() - 2 * eig.sqrt().sum()
    return max(0.0, float(dist))  # clamp tiny negative values caused by rounding


class InceptionFeatures(nn.Module):
    """Inception-v3 (ImageNet) pool features for FID. Input: (N, 3, H, W) float in [0, 1]."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


class VideoFeatures(nn.Module):
    """R3D-18 (Kinetics-400) features for FVD. Input: (N, T, 3, H, W) float in [0, 1]."""

    def __init__(self):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18
        self.net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.net.fc = nn.Identity()
        self.register_buffer("mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        n, t = x.shape[:2]
        x = F.interpolate(x.flatten(0, 1), size=(112, 112), mode="bilinear", align_corners=False)
        x = x.view(n, t, 3, 112, 112).permute(0, 2, 1, 3, 4)  # (N, 3, T, H, W)
        return self.net((x - self.mean) / self.std)
