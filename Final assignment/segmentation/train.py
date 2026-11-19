"""Train a single-frame segmentation model (Dice track baseline).

Run from the `Final assignment` folder:
    python -m segmentation.train --data-dir ./data/atlas120k --experiment-id my-run

The realtime track reuses `get_args_parser` and `main` with its own model (see `realtime/train.py`).
"""
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from atlas.data import SIZE, FrameDataset
from atlas.metrics import frame_dice
from atlas.training import init_run, log_masks, save_checkpoint, seed_everything


def get_args_parser(track="segmentation"):
    parser = ArgumentParser(f"Training script for the {track} track")
    parser.add_argument("--data-dir", type=str, default="./data/atlas120k", help="Path to ATLAS-120k")
    parser.add_argument("--size", type=int, nargs=2, default=list(SIZE), help="Training resolution (H W)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (cosine schedule)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--frame-stride", type=int, default=2, help="Use every n-th training frame per clip")
    parser.add_argument("--val-frame-stride", type=int, default=10, help="Use every n-th validation frame per clip")
    parser.add_argument("--num-workers", type=int, default=16, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default=f"{track}-baseline", help="Name of the W&B run and checkpoint folder")
    return parser


@torch.no_grad()
def validate(model, loader, criterion, device, step):
    """Return mean validation loss and per-frame Dice (at training resolution), and log examples."""
    model.eval()
    losses, dices = [], []
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
        losses.append(criterion(logits.float(), masks).item())
        preds = logits.argmax(1)
        for p, m in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            if (d := frame_dice(p, m)) is not None:
                dices.append(d)
        if i == 0:
            log_masks(images, preds, masks, step)
    return float(np.mean(losses)), float(np.mean(dices))


def main(args, model, track="segmentation"):
    out_dir = init_run(args, track)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_set = FrameDataset(args.data_dir, "train", tuple(args.size), augment=True, frame_stride=args.frame_stride)
    val_set = FrameDataset(args.data_dir, "val", tuple(args.size), frame_stride=args.val_frame_stride)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print(f"{len(train_set)} training frames, {len(val_set)} validation frames")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_dice, step = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            # bfloat16 mixed precision: faster on A100/H100, no loss scaling needed
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(images).float(), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch + 1}, step=step)

        val_loss, val_dice = validate(model, val_loader, criterion, device, step)
        wandb.log({"val_loss": val_loss, "val_dice": val_dice}, step=step)
        print(f"Epoch {epoch + 1}/{args.epochs}: val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, out_dir, "best")
    save_checkpoint(model, out_dir, "last")
    wandb.finish()


if __name__ == "__main__":
    from segmentation.model import Model

    args = get_args_parser().parse_args()
    main(args, Model(pretrained=True))
