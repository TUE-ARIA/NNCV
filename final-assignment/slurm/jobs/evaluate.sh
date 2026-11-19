#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --time=01:00:00
#
# Run a checkpoint exactly like the challenge server does, and score it with the server's metrics.
#
#   slurm/submit.sh evaluate <track> <checkpoint> [split]
#
#   slurm/submit.sh evaluate segmentation checkpoints/segmentation-baseline/best.pt
#   slurm/submit.sh evaluate generative checkpoints/generative-baseline/last.pt
#
# split: val (default) or test (the same data as the validation server).
# Predictions go to predictions/<track>-<split>/, the scores to the job log.
set -euo pipefail
source slurm/lib.sh
load_server

TRACK=${1:?usage: slurm/submit.sh evaluate <track> <checkpoint> [split]}
CHECKPOINT=${2:?usage: slurm/submit.sh evaluate <track> <checkpoint> [split]}
SPLIT=${3:-val}
check_track "$TRACK"
print_job_info "evaluate $TRACK $CHECKPOINT $SPLIT"

EVAL=data/eval/$SPLIT
OUT=predictions/$TRACK-$SPLIT
if [ "$TRACK" = generative ]; then
    INPUT=$EVAL/generative/masks
    SCORE_ARGS=(--gt-dir "$EVAL/generative" --reference-model data/reference_segmenter.pth)
else
    INPUT=$EVAL/segmentation/images
    SCORE_ARGS=(--gt-dir "$EVAL/segmentation/labels")
fi

in_container python -m submission.run --track "$TRACK" --checkpoint "$CHECKPOINT" --input-dir "$INPUT" --output-dir "$OUT"
in_container python evaluate.py "$TRACK" --pred-dir "$OUT" "${SCORE_ARGS[@]}"
