#!/bin/bash
# One-time setup on the cluster, run on the login node from the `final-assignment` folder:
#
#   bash slurm/setup.sh
#
# Pulls the container and links the shared course data into data/. Nothing is copied.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/lib.sh
load_server

if [ ! -f "$CONTAINER" ]; then
    apptainer pull "$CONTAINER" "$CONTAINER_IMAGE"
fi

mkdir -p data "$LOG_DIR"
ln -sfn "$SHARED_DIR/atlas120k" data/atlas120k                           # train / val / test splits
ln -sfn "$SHARED_DIR/eval" data/eval                                     # server-style val and test sets
ln -sfn "$SHARED_DIR/reference_segmenter.pth" data/reference_segmenter.pth  # ATLAS model for the generative mask Dice

ls -l data
echo "Setup done. Next: fill in .env, then: slurm/submit.sh train segmentation"
