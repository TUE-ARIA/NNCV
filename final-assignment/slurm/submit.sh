#!/bin/bash
# Submit a job to the cluster with the settings of this server (slurm/servers/<server>.env).
# Run it from the `final-assignment` folder.
#
#   slurm/submit.sh [sbatch options] <job> [job arguments]
#
#   slurm/submit.sh train segmentation
#   slurm/submit.sh train consistency --epochs 30 --experiment-id gru-30ep
#   slurm/submit.sh evaluate consistency checkpoints/gru-30ep/best.pt
#   slurm/submit.sh --time=08:00:00 train generative
#
# Jobs are the scripts in slurm/jobs/. sbatch options (starting with -) go before the job name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/lib.sh
load_server

sbatch_opts=()
while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do
    sbatch_opts+=("$1")
    shift
done
if [ $# -lt 1 ]; then
    echo "usage: slurm/submit.sh [sbatch options] <job> [job arguments]" >&2
    echo "jobs: $(ls slurm/jobs | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 1
fi
job=$1
shift
script="slurm/jobs/$job.sh"
[ -f "$script" ] || { echo "No job '$job' in slurm/jobs/" >&2; exit 1; }

opts=(
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --gpus="$GPUS"
    --cpus-per-task=$((GPUS * CPUS_PER_GPU))
    --chdir="$PWD"
    --output="$LOG_DIR/$job-%j.out"
    --export=ALL,SERVER="$SERVER"
)
if [ -n "${ACCOUNT:-}" ]; then opts+=(--account="$ACCOUNT"); fi

mkdir -p "$LOG_DIR"
sbatch "${opts[@]}" ${sbatch_opts[@]+"${sbatch_opts[@]}"} "$script" "$@"
echo "Log: $LOG_DIR/$job-<job id>.out   (follow it with: tail -f $LOG_DIR/$job-<job id>.out)"
