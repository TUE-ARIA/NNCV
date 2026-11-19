"""Convert an ATLAS-120k split into the folder layout the challenge server uses.

    python -m tools.make_eval_sets --data-dir ./data/atlas120k --split val --out-dir ./data/eval/val

Every frame is center-cropped to 3:2 and resized to 256x384, as in training. Creates:
    <out-dir>/segmentation/images/<clip_id>/frame_XXXXXX.png   RGB frames
    <out-dir>/segmentation/labels/<clip_id>/frame_XXXXXX.png   the matching masks with train IDs
    <out-dir>/generative/masks/<clip_id>/frame_XXXXXX.png      16-frame windows of train-ID masks
    <out-dir>/generative/frames/<clip_id>/frame_XXXXXX.png     the matching real frames
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from atlas.classes import to_train_ids
from atlas.data import ASPECT, SIZE, center_crop_aspect, contiguous_windows, find_clips

GEN_LENGTH = 16  # frames per generative clip


def load_pair(frame) -> tuple[Image.Image, Image.Image]:
    """Frame and train-ID mask, center-cropped to 3:2 and resized to SIZE."""
    img = center_crop_aspect(Image.open(frame.image).convert("RGB"), ASPECT).resize(SIZE[::-1], Image.BILINEAR)
    mask = center_crop_aspect(Image.open(frame.mask), ASPECT).resize(SIZE[::-1], Image.NEAREST)
    return img, Image.fromarray(to_train_ids(np.asarray(mask)))


def main():
    parser = ArgumentParser("Build server-style evaluation sets")
    parser.add_argument("--data-dir", type=str, default="./data/atlas120k")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gen-windows-per-clip", type=int, default=2, help="Generative windows taken from each clip")
    args = parser.parse_args()

    clips = find_clips(args.data_dir, args.split)
    seg, gen = args.out_dir / "segmentation", args.out_dir / "generative"
    for n, clip in enumerate(clips, 1):
        for sub in ("images", "labels"):
            (seg / sub / clip.id).mkdir(parents=True, exist_ok=True)
        for f in clip.frames:
            img, mask = load_pair(f)
            img.save(seg / "images" / clip.id / f"{f.image.stem}.png")
            mask.save(seg / "labels" / clip.id / f.mask.name)

        windows = contiguous_windows(clip.frames, GEN_LENGTH)
        # spread the selected windows over the clip
        picks = np.linspace(0, len(windows) - 1, min(args.gen_windows_per_clip, len(windows))).round().astype(int)
        for k in sorted(set(picks.tolist())):
            win_id = f"{clip.id}__w{k}"
            for sub in ("masks", "frames"):
                (gen / sub / win_id).mkdir(parents=True, exist_ok=True)
            for f in windows[k]:
                img, mask = load_pair(f)
                mask.save(gen / "masks" / win_id / f.mask.name)
                img.save(gen / "frames" / win_id / f"{f.image.stem}.png")
        print(f"[{n}/{len(clips)}] {clip.id}: {len(clip.frames)} frames, {len(set(picks.tolist()))} generative windows")


if __name__ == "__main__":
    main()
