"""Helpers used by the training scripts (seeding, logging, checkpoints)."""
import os
import random

import numpy as np
import torch
import wandb

from atlas.classes import colorize


def seed_everything(seed: int):
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_run(args, track: str) -> str:
    """Start a Weights & Biases run and return the checkpoint directory for this experiment."""
    wandb.init(project="5lsm0-atlas", group=track, name=args.experiment_id, config=vars(args))
    out_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def log_masks(images: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, step: int, n: int = 4):
    """Log the first `n` (image, prediction, label) triplets to W&B as one image per triplet."""
    rows = []
    for img, pred, label in zip(images[:n], preds[:n], labels[:n]):
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(wandb.Image(np.concatenate([img, colorize(pred), colorize(label)], axis=1),
                                caption="image | prediction | label"))
    wandb.log({"examples": rows}, step=step)


def save_checkpoint(model: torch.nn.Module, out_dir: str, name: str):
    """Save the model weights (state dict only) to `<out_dir>/<name>.pt`."""
    torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))
