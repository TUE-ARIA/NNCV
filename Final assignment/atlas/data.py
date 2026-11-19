"""Datasets for ATLAS-120k.

Expected layout (created by `download_data.sh`):

    <root>/<split>/<procedure>/<video>/<clip>/images/frame_000964.jpg
    <root>/<split>/<procedure>/<video>/<clip>/machine_masks/frame_000964.png

The number in the filename is the frame index in the 15 fps video. Frames inside a clip
can have gaps, so consecutive files are not always consecutive video frames.

Every frame is center-cropped to the largest 3:2 region (`ASPECT`) and resized to `SIZE`
(256x384), in all tracks and on the server, so nothing is stretched. Images are returned as float tensors
in [0, 1] (normalisation happens inside the models), masks as int64 tensors with train IDs.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from atlas.classes import to_train_ids

# Videos range from 4:3 (480x640) to 16:9 (1080x1920); 3:2 loses the least of both.
ASPECT = 3 / 2          # width / height of the center crop used everywhere
SIZE = (256, 384)       # resolution of all frames and masks (height, width), also 3:2


@dataclass
class Frame:
    index: int          # frame index in the 15 fps video
    image: Path
    mask: Path


@dataclass
class Clip:
    id: str             # "<procedure>__<video>__<clip>"
    frames: list[Frame]


def frame_index(path: Path) -> int:
    """Frame index from a filename like `frame_000964.png`."""
    return int(path.stem.split("_")[-1])


def find_clips(root: str | Path, split: str) -> list[Clip]:
    """List all clips of a split, with frames sorted in time. Frames without an image are skipped."""
    clips = []
    for mask_dir in sorted(Path(root, split).glob("*/*/*/machine_masks")):
        clip_dir = mask_dir.parent
        frames = []
        for mask in sorted(mask_dir.glob("*.png"), key=frame_index):
            image = next((p for p in (clip_dir / "images" / f"{mask.stem}{ext}" for ext in (".jpg", ".png")) if p.exists()), None)
            if image is not None:
                frames.append(Frame(frame_index(mask), image, mask))
        if frames:
            procedure, video, clip = clip_dir.parts[-3:]
            clips.append(Clip(f"{procedure}__{video}__{clip}", frames))
    if not clips:
        raise FileNotFoundError(f"No clips found in {Path(root, split)}. Did you run download_data.sh?")
    return clips


def contiguous_windows(frames: list[Frame], length: int, max_gap: int = 1, stride: int | None = None) -> list[list[Frame]]:
    """Split a clip into windows of `length` frames where neighbours are at most `max_gap` video frames apart."""
    stride = stride or length
    runs, run = [], [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        if cur.index - prev.index <= max_gap:
            run.append(cur)
        else:
            runs.append(run)
            run = [cur]
    runs.append(run)
    return [r[s:s + length] for r in runs for s in range(0, len(r) - length + 1, stride)]


def center_crop_aspect(img: Image.Image, aspect: float = ASPECT) -> Image.Image:
    """Largest centred crop with width / height = `aspect`."""
    w, h = img.size
    cw, ch = (round(h * aspect), h) if w / h > aspect else (w, round(w / aspect))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def load_frame(frame: Frame, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one frame as (image (3, H, W) float in [0, 1], mask (H, W) int64 train IDs).

    The frame is center-cropped to 3:2 and resized to `size` (height, width).
    """
    img = center_crop_aspect(Image.open(frame.image).convert("RGB"))
    mask = center_crop_aspect(Image.open(frame.mask))
    img = img.resize(size[::-1], Image.BILINEAR)
    mask = mask.resize(size[::-1], Image.NEAREST)
    img = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    mask = torch.from_numpy(to_train_ids(np.asarray(mask)).astype(np.int64))
    return img, mask


class FrameDataset(Dataset):
    """Single frames. Returns (image (3, H, W), mask (H, W)).

    Args:
        root: dataset root, e.g. `./data/atlas120k`.
        split: "train", "val" or "test".
        size: (height, width) to resize the 3:2 crop to.
        augment: random horizontal flip.
        frame_stride: keep every n-th frame of each clip, to make epochs shorter.
    """

    def __init__(self, root, split, size=SIZE, augment=False, frame_stride=1):
        self.frames = [f for clip in find_clips(root, split) for f in clip.frames[::frame_stride]]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        img, mask = load_frame(self.frames[i], self.size)
        if self.augment and random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        return img, mask


class ClipDataset(Dataset):
    """Short windows of consecutive frames. Returns (images (T, 3, H, W), masks (T, H, W)).

    Args:
        length: frames per window (T).
        max_gap: maximum gap in video frames between neighbouring frames in a window.
        stride: step between window starts; defaults to `length` (no overlap).
        Other arguments as in `FrameDataset`. The same flip is applied to every frame of a window.
    """

    def __init__(self, root, split, length=8, size=SIZE, augment=False, max_gap=1, stride=None):
        self.windows = [w for clip in find_clips(root, split) for w in contiguous_windows(clip.frames, length, max_gap, stride)]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        imgs, masks = zip(*(load_frame(f, self.size) for f in self.windows[i]))
        imgs, masks = torch.stack(imgs), torch.stack(masks)
        if self.augment and random.random() < 0.5:
            imgs, masks = imgs.flip(-1), masks.flip(-1)
        return imgs, masks


def one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W) int mask -> (B, C, H, W) float one-hot."""
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
