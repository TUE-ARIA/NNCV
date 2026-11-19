"""Train the consistency baseline (U-Net + ConvGRU) on short clips.

Run from the `Final assignment` folder:
    python -m consistency.train --data-dir ./data/atlas120k --experiment-id my-run

Tip: start from a trained Dice-track model with `--init-from checkpoints/<run>/best.pt`.
"""
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from atlas.data import SIZE, ClipDataset
from atlas.metrics import ClipScorer, summarize
from atlas.training import init_run, log_masks, save_checkpoint, seed_everything
from consistency.model import Model


def get_args_parser():
    parser = ArgumentParser("Training script for the consistency track")
    parser.add_argument("--data-dir", type=str, default="./data/atlas120k", help="Path to ATLAS-120k")
    parser.add_argument("--size", type=int, nargs=2, default=list(SIZE), help="Training resolution (H W)")
    parser.add_argument("--clip-length", type=int, default=8, help="Frames per training clip")
    parser.add_argument("--batch-size", type=int, default=6, help="Clips per batch")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (cosine schedule)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--init-from", type=str, default=None, help="Optional Dice-track checkpoint for the U-Net")
    parser.add_argument("--num-workers", type=int, default=16, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="consistency-baseline", help="Name of the W&B run and checkpoint folder")
    return parser


@torch.no_grad()
def validate(model, loader, device, step):
    """Dice, Change Dice and consistency score on validation windows (at training resolution)."""
    model.eval()
    scorers = []
    for i, (clips, masks) in enumerate(loader):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            preds = model.forward_clip(clips.to(device)).argmax(2).cpu().numpy()
        for pred_clip, gt_clip in zip(preds, masks.numpy()):
            scorer = ClipScorer()
            for t, (p, g) in enumerate(zip(pred_clip, gt_clip)):
                scorer.add(t, p, g)  # windows are contiguous, so t works as frame index
            scorers.append(scorer)
        if i == 0:
            log_masks(clips[0], torch.from_numpy(preds[0]), masks[0], step)  # frames of one clip
    return summarize(scorers)


def main(args):
    out_dir = init_run(args, "consistency")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(pretrained=True)
    if args.init_from:
        model.unet.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=True))
    model = model.to(device)

    size = tuple(args.size)
    train_set = ClipDataset(args.data_dir, "train", args.clip_length, size, augment=True)
    val_set = ClipDataset(args.data_dir, "val", 16, size)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print(f"{len(train_set)} training clips, {len(val_set)} validation clips")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_score, step = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        for clips, masks in train_loader:
            clips, masks = clips.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.forward_clip(clips)                     # (B, T, C, H, W)
                # per-frame cross-entropy; add a temporal loss here to improve consistency
                loss = criterion(logits.flatten(0, 1).float(), masks.flatten(0, 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch + 1}, step=step)

        scores = validate(model, val_loader, device, step)
        wandb.log({f"val_{k}": v for k, v in scores.items()}, step=step)
        print(f"Epoch {epoch + 1}/{args.epochs}: dice={scores['dice']:.4f} "
              f"change_dice={scores['change_dice']:.4f} consistency_score={scores['consistency_score']:.4f}")
        if scores["consistency_score"] > best_score:
            best_score = scores["consistency_score"]
            save_checkpoint(model, out_dir, "best")
    save_checkpoint(model, out_dir, "last")
    wandb.finish()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
