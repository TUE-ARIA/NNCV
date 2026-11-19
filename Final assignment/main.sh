#!/bin/bash
# Runs inside the container.
#   Train:    bash main.sh <track>
#   Evaluate: bash main.sh evaluate <track> <checkpoint>   (scores the val split like the server)
# Change the hyperparameters of your track below; see `python -m <track>.train --help` for all options.
TRACK=${1:-segmentation}

if [ "$TRACK" = "evaluate" ]; then
  TRACK=$2; CHECKPOINT=$3
  if [ "$TRACK" = "generative" ]; then
    INPUT=data/eval/val/generative/masks; GT=data/eval/val/generative; EXTRA="--reference-model data/reference_segmenter.pth"
  else
    INPUT=data/eval/val/segmentation/images; GT=data/eval/val/segmentation/labels; EXTRA=""
  fi
  python3 -m submission.run --track "$TRACK" --checkpoint "$CHECKPOINT" --input-dir $INPUT --output-dir "predictions/$TRACK"
  python3 evaluate.py "$TRACK" --pred-dir "predictions/$TRACK" --gt-dir $GT $EXTRA
  exit
fi

wandb login

case $TRACK in
  segmentation)
    python3 -m segmentation.train \
      --data-dir ./data/atlas120k \
      --batch-size 32 \
      --epochs 20 \
      --lr 3e-4 \
      --experiment-id "segmentation-baseline"
    ;;
  consistency)
    python3 -m consistency.train \
      --data-dir ./data/atlas120k \
      --clip-length 8 \
      --batch-size 6 \
      --epochs 20 \
      --experiment-id "consistency-baseline"
      # --init-from checkpoints/segmentation-baseline/best.pt
    ;;
  realtime)
    python3 -m realtime.train \
      --data-dir ./data/atlas120k \
      --batch-size 64 \
      --epochs 20 \
      --lr 1e-3 \
      --experiment-id "realtime-baseline"
    ;;
  generative)
    python3 -m generative.train \
      --data-dir ./data/atlas120k \
      --batch-size 16 \
      --epochs 10 \
      --experiment-id "generative-baseline"
    ;;
  *)
    echo "Unknown track '$TRACK'. Use: segmentation, consistency, realtime or generative." && exit 1
    ;;
esac
