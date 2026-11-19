"""Train the pix2pix baseline (mask -> frame), one frame at a time.

Run from the `Final assignment` folder:
    python -m generative.train --data-dir ./data/atlas120k --experiment-id my-run

Frames are center-cropped to 3:2 and resized to 256x384, like everywhere in this assignment.
"""
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from atlas.classes import NUM_CLASSES, colorize
from atlas.data import SIZE, FrameDataset, one_hot
from atlas.training import init_run, save_checkpoint, seed_everything
from generative.model import Discriminator, Generator


def get_args_parser():
    parser = ArgumentParser("Training script for the generative track")
    parser.add_argument("--data-dir", type=str, default="./data/atlas120k", help="Path to ATLAS-120k")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate (generator and discriminator)")
    parser.add_argument("--l1-weight", type=float, default=100.0, help="Weight of the L1 reconstruction loss")
    parser.add_argument("--frame-stride", type=int, default=2, help="Use every n-th training frame per clip")
    parser.add_argument("--num-workers", type=int, default=16, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="generative-baseline", help="Name of the W&B run and checkpoint folder")
    return parser


def log_samples(masks, fakes, reals, step, n=4):
    """Log (mask | generated | real) triplets to W&B."""
    to_img = lambda x: ((x.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    rows = [wandb.Image(np.concatenate([colorize(m), to_img(f), to_img(r)], axis=1), caption="mask | generated | real")
            for m, f, r in zip(masks[:n], fakes[:n], reals[:n])]
    wandb.log({"samples": rows}, step=step)


def main(args):
    out_dir = init_run(args, "generative")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen, disc = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = nn.MSELoss()  # least-squares GAN loss, more stable than the original BCE loss

    train_set = FrameDataset(args.data_dir, "train", SIZE, augment=True, frame_stride=args.frame_stride)
    loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    print(f"{len(train_set)} training frames")

    step = 0
    for epoch in range(args.epochs):
        gen.train()
        for images, masks in loader:
            real = images.to(device) * 2 - 1                      # [0, 1] -> [-1, 1] to match tanh
            cond = one_hot(masks.to(device), NUM_CLASSES)
            fake = gen(cond)

            # Discriminator: real pairs -> 1, generated pairs -> 0
            pred_real, pred_fake = disc(cond, real), disc(cond, fake.detach())
            loss_d = 0.5 * (gan_loss(pred_real, torch.ones_like(pred_real)) + gan_loss(pred_fake, torch.zeros_like(pred_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Generator: fool the discriminator and stay close to the real frame
            pred_fake = disc(cond, fake)
            loss_gan = gan_loss(pred_fake, torch.ones_like(pred_fake))
            loss_l1 = F.l1_loss(fake, real)
            loss_g = loss_gan + args.l1_weight * loss_l1
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            step += 1
            if step % 50 == 0:
                wandb.log({"loss_d": loss_d.item(), "loss_g_gan": loss_gan.item(), "loss_g_l1": loss_l1.item(), "epoch": epoch + 1}, step=step)
            if step % 1000 == 0:
                log_samples(masks, fake.detach(), real, step)

        print(f"Epoch {epoch + 1}/{args.epochs} done")
        save_checkpoint(gen, out_dir, "last")  # only the generator is needed for submission
    wandb.finish()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
