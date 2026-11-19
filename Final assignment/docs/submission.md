# Submitting

You submit a **Docker image** (exported as a `.tar` file) that contains your code and weights. The server runs it on clips it has never shown you and returns your scores.

---

## 1. How the server runs your code

The image's entrypoint is [submission/run.py](../submission/run.py). It:

1. builds your track's `Predictor` from `<track>/predictor.py` with the weights at `/app/model.pt`,
2. reads clips from `/data/<clip_id>/` and writes results to `/output/<clip_id>/`, with the same file names,
3. passes segmentation frames to `step(frame)` **one at a time and in temporal order** (calling `reset()` before each clip) and times every call,
4. passes each generative clip of 16 masks to `generate(masks)`.

The server uses **its own copy** of `submission/run.py` and `atlas/metrics.py`, so changes to those files have no effect. Put all your changes in your track's folder (or in new files).

The image has **no internet access** when it runs. Everything it needs must be in the image. Build your model with `pretrained=False` in `predictor.py` and load your own checkpoint.

## 2. Evaluate locally first

The server-style evaluation sets are already in `data/eval/` (the `val` split for you to use; `test` is the same data the validation server uses):

```
data/eval/val/segmentation/images/<clip_id>/frame_*.png     input frames (256×384)
data/eval/val/segmentation/labels/<clip_id>/frame_*.png     ground truth (train IDs)
data/eval/val/generative/masks/<clip_id>/frame_*.png        input masks, 16 per clip
data/eval/val/generative/frames/<clip_id>/frame_*.png       real frames
```

Run your predictor exactly as the server does, then score it:

```bash
python -m submission.run --track consistency --checkpoint checkpoints/my-run/best.pt \
    --input-dir data/eval/val/segmentation/images --output-dir predictions/consistency
python evaluate.py consistency --pred-dir predictions/consistency --gt-dir data/eval/val/segmentation/labels
```

On Snellius, one job runs both commands for you:

```bash
slurm/submit.sh evaluate consistency checkpoints/my-run/best.pt
```

The scores are in the `slurm-<job_id>.out` file. Don't run evaluation on the login node.

## 3. Build the image

Copy the checkpoint you want to submit to `Final assignment/model.pt`, then build from the `Final assignment` folder with your track as build argument:

```bash
cp checkpoints/my-run/best.pt model.pt
docker build --build-arg TRACK=consistency -t nncv-consistency .
```

`TRACK` is one of `segmentation`, `consistency`, `realtime` or `generative`. If you need extra Python packages, add them to the `pip install` line in the [Dockerfile](../Dockerfile).

## 4. Test the image

Use a few clips from `data/eval/val` (copy them to your laptop):

```bash
docker run --rm --gpus all -v "$(pwd)/local_data:/data" -v "$(pwd)/local_output:/output" nncv-consistency
```

On Windows PowerShell, use `${PWD}` instead of `$(pwd)`. Leave out `--gpus all` if you don't have an NVIDIA GPU. It will be slow, but it shows whether the image works.

## 5. Export and submit

```bash
docker save -o submission_consistency.tar nncv-consistency
```

Upload the `.tar` to the server of your track (TU/e network or VPN only):

| Track | Validation server (until TBD) | Hidden test server (final week) |
|---|---|---|
| Dice | TBD | TBD |
| Consistency | TBD | TBD |
| Real-time | TBD | TBD |
| Generative | TBD | TBD |

- The validation servers score on the `test` split and show a leaderboard.
- The hidden test servers open in the final week, allow **TBD submissions per track**, and return the scores you report in your paper.
- Every submission has a time limit of **TBD minutes**.

## 6. Something doesn't work?
See the [FAQ](faq.md#submission) for the most common problems.

Next: [Report & grading](report.md).
