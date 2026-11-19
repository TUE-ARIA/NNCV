"""Score predictions locally with the same metrics as the challenge server.

Run from the `final-assignment` folder after `python -m submission.run ...`:

    python evaluate.py segmentation --pred-dir predictions/segmentation --gt-dir data/eval/val/segmentation/labels
    python evaluate.py consistency  --pred-dir predictions/consistency  --gt-dir data/eval/val/segmentation/labels
    python evaluate.py realtime     --pred-dir predictions/realtime     --gt-dir data/eval/val/segmentation/labels
    python evaluate.py generative   --pred-dir predictions/generative   --gt-dir data/eval/val/generative \
                                    --reference-model data/reference_segmenter.pth

The leaderboard ranks on: segmentation -> dice, consistency -> consistency_score,
realtime -> dice (only if fps >= MIN_FPS on the server GPU),
generative -> fvd (lower is better, only if mask_dice >= MASK_DICE_FLOOR x the mask Dice of real frames).
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from atlas.metrics import ClipScorer, InceptionFeatures, VideoFeatures, frame_dice, frechet_distance, summarize
from submission.run import list_frames

MIN_FPS = 25.0          # real-time threshold (batch size 1, measured by submission/run.py)
MASK_DICE_FLOOR = 0.7   # generated frames must reach 70% of the reference segmenter's Dice on real frames


def load_clip(clip_dir: Path) -> tuple[list[int], np.ndarray]:
    """Frame indices and stacked images of one clip folder."""
    paths = list_frames(clip_dir)
    return [int(p.stem.split("_")[-1]) for p in paths], np.stack([np.asarray(Image.open(p)) for p in paths])


def evaluate_segmentation(pred_dir: Path, gt_dir: Path) -> dict:
    scorers = []
    for gt_clip in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        scorer = ClipScorer()
        for gt_path in list_frames(gt_clip):
            pred_path = pred_dir / gt_clip.name / gt_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction {pred_path}")
            scorer.add(int(gt_path.stem.split("_")[-1]), np.asarray(Image.open(pred_path)), np.asarray(Image.open(gt_path)))
        scorers.append(scorer)
    return summarize(scorers)


@torch.no_grad()
def evaluate_generative(pred_dir: Path, gt_dir: Path, reference_model: str | None, batch_size: int = 8) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_net, video_net = InceptionFeatures().to(device), VideoFeatures().to(device)
    segmenter = None
    if reference_model:
        from atlas.reference import ReferenceSegmenter  # ATLAS model, needs `transformers`
        segmenter = ReferenceSegmenter(reference_model, device)

    clip_ids = sorted(p.name for p in (gt_dir / "frames").iterdir() if p.is_dir())
    feats = {"fake_img": [], "real_img": [], "fake_vid": [], "real_vid": []}
    mask_dices = {"fake": [], "real": []}
    for i in range(0, len(clip_ids), batch_size):
        ids = clip_ids[i:i + batch_size]
        to_tensor = lambda clips: torch.from_numpy(np.stack(clips)).to(device).permute(0, 1, 4, 2, 3).float() / 255
        fake = to_tensor([load_clip(pred_dir / c)[1] for c in ids])           # (B, T, 3, H, W)
        real = to_tensor([load_clip(gt_dir / "frames" / c)[1] for c in ids])
        feats["fake_img"].append(image_net(fake.flatten(0, 1)).cpu())
        feats["real_img"].append(image_net(real.flatten(0, 1)).cpu())
        feats["fake_vid"].append(video_net(fake).cpu())
        feats["real_vid"].append(video_net(real).cpu())
        if segmenter is not None:
            masks = np.stack([load_clip(gt_dir / "masks" / c)[1] for c in ids]).reshape(-1, *fake.shape[-2:])
            for name, frames in (("fake", fake), ("real", real)):
                preds = segmenter(frames.flatten(0, 1)).argmax(1).cpu().numpy()
                mask_dices[name] += [d for p, m in zip(preds, masks) if (d := frame_dice(p, m)) is not None]

    feats = {k: torch.cat(v) for k, v in feats.items()}
    scores = {
        "fvd": frechet_distance(feats["fake_vid"], feats["real_vid"]),
        "fid": frechet_distance(feats["fake_img"], feats["real_img"]),
        "num_clips": len(clip_ids),
    }
    if segmenter is not None:
        scores["mask_dice"] = float(np.mean(mask_dices["fake"]))
        scores["mask_dice_real"] = float(np.mean(mask_dices["real"]))  # what real frames reach
        scores["meets_mask_dice_floor"] = scores["mask_dice"] >= MASK_DICE_FLOOR * scores["mask_dice_real"]
    return scores


def main():
    parser = ArgumentParser("Evaluate predictions")
    parser.add_argument("track", choices=["segmentation", "consistency", "realtime", "generative"])
    parser.add_argument("--pred-dir", type=Path, required=True, help="Output folder of submission/run.py")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Label folder (segmentation tracks) or folder with frames/ and masks/ (generative)")
    parser.add_argument("--reference-model", type=str, default=None, help="Generative only: ATLAS reference weights for mask Dice")
    args = parser.parse_args()

    if args.track == "generative":
        scores = evaluate_generative(args.pred_dir, args.gt_dir, args.reference_model)
    else:
        scores = evaluate_segmentation(args.pred_dir, args.gt_dir)
        timing_file = args.pred_dir / "timing.json"
        if timing_file.exists():
            scores["fps"] = json.loads(timing_file.read_text())["fps"]
            if args.track == "realtime":
                scores["meets_fps_threshold"] = scores["fps"] >= MIN_FPS
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
