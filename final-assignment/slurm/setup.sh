#!/bin/bash
# One-time setup on the cluster, run on the login node from the `final-assignment` folder:
#
#   bash slurm/setup.sh                  links the course data and the containers, creates your .env
#   bash slurm/setup.sh transformers     also gets this container, if the course folder doesn't have it
#
# The containers are links to the course folder when the staff have put them there. Otherwise the
# images you ask for (default: pytorch) are pulled, each into its own file, so nothing is overwritten.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/lib.sh
load_server

wanted=${*:-$IMAGE}
for image in $wanted; do check_image "$image"; done

mkdir -p data "$LOG_DIR" "$CONTAINER_DIR" wandb
ln -sfn "$SHARED_DIR/atlas120k" data/atlas120k                           # train / val / test splits
ln -sfn "$SHARED_DIR/eval" data/eval                                     # server-style val and test sets
ln -sfn "$SHARED_DIR/reference_segmenter.pth" data/reference_segmenter.pth  # ATLAS model for the generative mask Dice

for image in $IMAGES; do
    sif=$(container_of "$image")
    shared="${SHARED_CONTAINERS:-}/nncv-$image.sif"
    if [ -n "${SHARED_CONTAINERS:-}" ] && [ -f "$shared" ]; then
        ln -sfn "$shared" "$sif"                                         # no download, no storage used
    elif [ ! -e "$sif" ] && [[ " $wanted " == *" $image "* ]]; then
        apptainer pull "$sif" "$IMAGE_PREFIX$image:$IMAGE_TAG"
    fi
done

if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env                                                       # only you can read your API key
fi

ls -l data "$CONTAINER_DIR"
echo "Setup done. Next: put your W&B API key in .env, then: slurm/submit.sh train segmentation"
