#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

# Usage:
#   sbatch jobscript_slurm.sh <track>                          train (segmentation, consistency, realtime or generative)
#   sbatch jobscript_slurm.sh evaluate <track> <checkpoint>    run and score a checkpoint on the val split
srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh "$@"
