#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=04:00:00
#
# Train a model for one track.
#
#   slurm/submit.sh train <track> [--flag value ...]
#
#   slurm/submit.sh train segmentation
#   slurm/submit.sh train consistency --init-from checkpoints/segmentation-baseline/best.pt
#   slurm/submit.sh train realtime --batch-size 32 --lr 5e-4 --experiment-id realtime-bs32
#
# Flags go to `python -m <track>.train`; without flags its defaults are used.
# See all flags with: apptainer exec containers/nncv-pytorch.sif python -m <track>.train --help
# Checkpoints go to checkpoints/<experiment-id>/, training curves to Weights & Biases.
set -euo pipefail
source slurm/lib.sh
load_server

TRACK=${1:?usage: slurm/submit.sh train <track> [--flag value ...]}
shift
check_track "$TRACK"
print_job_info "train $TRACK $*"

in_container python -m "$TRACK.train" --data-dir data/atlas120k "$@"
