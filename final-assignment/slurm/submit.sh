#!/bin/bash
# Submit a job to the cluster with the settings of this server (slurm/servers/<server>.env).
# Run it from the `final-assignment` folder.
#
#   slurm/submit.sh [--image <image>] [sbatch options] <job> [job arguments]
#
#   slurm/submit.sh train segmentation
#   slurm/submit.sh train consistency --epochs 30 --experiment-id gru-30ep
#   slurm/submit.sh evaluate consistency checkpoints/gru-30ep/best.pt
#   slurm/submit.sh --time=08:00:00 train generative
#   slurm/submit.sh --image transformers train segmentation --experiment-id dinov2
#
# Jobs are the scripts in slurm/jobs/. Options (starting with -) go before the job name.
# --image chooses the container (default: pytorch); everything else goes to sbatch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/lib.sh
load_server

sbatch_opts=()
while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do
    case "$1" in
        --image) IMAGE=${2:?--image needs a name}; shift 2 ;;
        --image=*) IMAGE=${1#--image=}; shift ;;
        *) sbatch_opts+=("$1"); shift ;;
    esac
done
if [ $# -lt 1 ]; then
    echo "usage: slurm/submit.sh [--image <image>] [sbatch options] <job> [job arguments]" >&2
    echo "jobs: $(ls slurm/jobs | sed 's/\.sh$//' | tr '\n' ' ')  images: $IMAGES" >&2
    exit 1
fi
job=$1
shift
script="slurm/jobs/$job.sh"
[ -f "$script" ] || { echo "No job '$job' in slurm/jobs/" >&2; exit 1; }

# catch a missing setup here, not after the job has waited in the queue
require_container "$IMAGE"
[ -f .env ] || { echo "No .env yet: run 'bash slurm/setup.sh' first." >&2; exit 1; }
if [ "$job" = train ] && ! grep -Eq '^WANDB_API_KEY=[^[:space:]]+' .env; then
    echo "Put your W&B API key in .env (WANDB_API_KEY=...), see https://wandb.ai/authorize" >&2
    exit 1
fi

opts=(
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --gpus="$GPUS"
    --cpus-per-task=$((GPUS * CPUS_PER_GPU))
    --chdir="$PWD"
    --output="$LOG_DIR/$job-%j.out"
    --export=ALL,SERVER="$SERVER",IMAGE="$IMAGE"
)
if [ -n "${ACCOUNT:-}" ]; then opts+=(--account="$ACCOUNT"); fi

mkdir -p "$LOG_DIR"
sbatch "${opts[@]}" ${sbatch_opts[@]+"${sbatch_opts[@]}"} "$script" "$@"
echo "Log: $LOG_DIR/$job-<job id>.out   (follow it with: tail -f $LOG_DIR/$job-<job id>.out)"
