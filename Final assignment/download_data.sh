#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=00:30:00

# One-time setup: pulls the training container and links the shared course data.
# Nothing is copied: the data lives in a shared folder on Snellius.

SHARED_DIR=/projects/0/TODO_5lsm0   # TODO(staff): shared course folder on Snellius

# Training container (PyTorch, torchvision, wandb, ...)
apptainer pull container.sif docker://cclaess/5lsm0:v1

mkdir -p data
ln -sfn "$SHARED_DIR/atlas120k" data/atlas120k                         # train / val / test splits
ln -sfn "$SHARED_DIR/eval" data/eval                                   # server-style val and test sets
ln -sfn "$SHARED_DIR/reference_segmenter.pth" data/reference_segmenter.pth  # ATLAS model for the generative mask Dice

ls -l data
