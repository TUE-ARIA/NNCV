"""Runs a submission on a folder of clips. This is the entrypoint of the Docker image.

Do not change this file: the server replaces it with its own copy. Change your track's `predictor.py` instead.

Segmentation tracks (segmentation, consistency, realtime):
    <input-dir>/<clip_id>/frame_000964.png  ->  <output-dir>/<clip_id>/frame_000964.png  (train IDs)
    Frames are passed to `Predictor.step` one by one, in temporal order. Each call is timed and the
    result is written to <output-dir>/timing.json.

Generative track:
    <input-dir>/<clip_id>/frame_000964.png (mask)  ->  <output-dir>/<clip_id>/frame_000964.png (RGB)

Local example, from the `Final assignment` folder:
    python -m submission.run --track segmentation --checkpoint model.pt \
        --input-dir data/eval/val/segmentation/images --output-dir predictions/segmentation
"""
import importlib
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TRACKS = ("segmentation", "consistency", "realtime", "generative")
WARMUP_FRAMES = 10  # the first frames are not timed (CUDA initialisation, autotuning)


def list_frames(clip_dir: Path) -> list[Path]:
    """Frames of a clip sorted by frame index."""
    files = [p for p in clip_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def run_segmentation(predictor, clips: list[Path], output_dir: Path, device):
    timed_frames, timed_seconds, seen = 0, 0.0, 0
    for clip in clips:
        out_clip = output_dir / clip.name
        out_clip.mkdir(parents=True, exist_ok=True)
        predictor.reset()
        for path in list_frames(clip):
            frame = np.array(Image.open(path).convert("RGB"))
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            mask = predictor.step(frame)
            mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            seen += 1
            if seen > WARMUP_FRAMES:
                timed_frames, timed_seconds = timed_frames + 1, timed_seconds + elapsed
            if mask.shape != frame.shape[:2]:
                raise ValueError(f"{path}: mask shape {mask.shape} must equal frame shape {frame.shape[:2]}")
            Image.fromarray(mask.astype(np.uint8)).save(out_clip / f"{path.stem}.png")
    fps = timed_frames / timed_seconds if timed_seconds else 0.0
    timing = {"frames": seen, "timed_frames": timed_frames, "seconds": timed_seconds, "fps": fps,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"Processed {seen} frames at {fps:.1f} FPS on {timing['device']}")


def run_generative(predictor, clips: list[Path], output_dir: Path):
    for clip in clips:
        paths = list_frames(clip)
        masks = np.stack([np.asarray(Image.open(p)) for p in paths]).astype(np.uint8)
        frames = np.asarray(predictor.generate(masks))
        if frames.shape != (*masks.shape, 3) or frames.dtype != np.uint8:
            raise ValueError(f"{clip.name}: expected uint8 frames of shape {(*masks.shape, 3)}, got {frames.dtype} {frames.shape}")
        out_clip = output_dir / clip.name
        out_clip.mkdir(parents=True, exist_ok=True)
        for path, frame in zip(paths, frames):
            Image.fromarray(frame).save(out_clip / f"{path.stem}.png")
    print(f"Generated {len(clips)} clips")


def main():
    parser = ArgumentParser("Run a submission")
    parser.add_argument("--track", default=os.environ.get("TRACK"), choices=TRACKS, help="Defaults to $TRACK (set in the Dockerfile)")
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--checkpoint", default="/app/model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = importlib.import_module(f"{args.track}.predictor").Predictor(args.checkpoint, device)
    clips = sorted(p for p in Path(args.input_dir).iterdir() if p.is_dir())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Track '{args.track}': {len(clips)} clips on {device}")

    with torch.inference_mode():
        if args.track == "generative":
            run_generative(predictor, clips, output_dir)
        else:
            run_segmentation(predictor, clips, output_dir, device)


if __name__ == "__main__":
    main()
