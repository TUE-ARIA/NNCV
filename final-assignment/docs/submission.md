# Submitting

You submit a **Docker image**, exported as a `.tar` file, that contains your code and your weights. The server runs it on clips you haven't seen and returns your scores.

## 1. How the server runs your code

The entrypoint of the image is [submission/run.py](../submission/run.py). It:

1. builds the `Predictor` from your track's `predictor.py`, with the weights at `/app/model.pt`,
2. reads the clips from `/data/<clip_id>/` and writes your results to `/output/<clip_id>/`, with the same file names,
3. for the segmentation tracks, calls `reset()` at the start of every clip and `step(frame)` for every frame in order, timing each call,
4. for the generative track, calls `generate(masks)` once per clip of 16 masks.

The server uses its own copies of `submission/run.py` and `atlas/metrics.py`, so changing them has no effect. Put your changes in your track's folder or in new files.

> [!WARNING]
> The image has **no internet access** on the server. Everything your predictor needs must be inside the image: build your model with `pretrained=False` in `predictor.py`, and load your own checkpoint.

## 2. Evaluate on Snellius first

An evaluate job runs your predictor exactly as the server does, on the `val` split, and scores it with the server's metrics:

```bash
slurm/submit.sh evaluate consistency checkpoints/my-run/best.pt
```

The scores are at the end of the job log, `slurm/logs/evaluate-<job id>.out`, and the predictions are in `predictions/<track>-val/`. The evaluation sets are already prepared in `data/eval/`:

```text
data/eval/val/segmentation/images/<clip_id>/frame_*.png   input frames (256×384)
data/eval/val/segmentation/labels/<clip_id>/frame_*.png   ground truth (train IDs)
data/eval/val/generative/masks/<clip_id>/frame_*.png      input masks, 16 per clip
data/eval/val/generative/frames/<clip_id>/frame_*.png     real frames
```

`data/eval/test` has the same layout and holds the data of the validation server.

## 3. Build the image

On your laptop, copy the checkpoint you want to submit to `final-assignment/model.pt`, and build the image from the `final-assignment` folder with your track as build argument:

```bash
scp <username>@snellius.surf.nl:NNCV/final-assignment/checkpoints/my-run/best.pt model.pt
docker build --build-arg TRACK=consistency -t nncv-consistency .
```

`TRACK` is `segmentation`, `consistency`, `realtime` or `generative`. If your predictor needs extra Python packages, add them to the `pip install` line in the [Dockerfile](../Dockerfile).

## 4. Test the image

Copy one or two clips from `data/eval/val` to a folder `local_data/` on your laptop, and run the image on them:

```bash
docker run --rm --gpus all -v "$(pwd)/local_data:/data" -v "$(pwd)/local_output:/output" nncv-consistency
```

On Windows PowerShell, write `${PWD}` instead of `$(pwd)`. Without an NVIDIA GPU, leave out `--gpus all`: it is slow, but it shows whether the image works.

## 5. Export and submit

```bash
docker save -o submission_consistency.tar nncv-consistency
```

Upload the `.tar` to the server of your track. The servers are only reachable from the TU/e network or over VPN.

| Track | Validation server (until TBD) | Hidden test server (final week) |
| --- | --- | --- |
| Dice | TBD | TBD |
| Consistency | TBD | TBD |
| Real-time | TBD | TBD |
| Generative | TBD | TBD |

- The **validation servers** score on the `test` split and show a leaderboard. These are the scores you report in your paper.
- The **hidden test servers** open in the final week and take **TBD submissions per track**. They don't return scores: the ranking is announced on the challenge day, and the bonus points are based on it.
- Every submission has a time limit of **TBD minutes**.

Something doesn't work? See the [FAQ](faq.md#submission).

Next: [Report, pitch & grading](report.md).
