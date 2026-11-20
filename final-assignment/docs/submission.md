# Submitting

You submit a **zip with your code and weights** on the challenge website. The server runs it in one of the course's environments, on its GPU and without internet access, on clips you haven't seen, and puts your scores on the leaderboard of your track.

## 1. How the server runs your code

Your code runs inside a ready-made Docker image, the *base image*, that you choose. The server adds the fixed harness [submission/run.py](../submission/run.py), which:

1. builds the `Predictor` from your track's `predictor.py` with your weights,
2. reads the clips from `/data/<clip_id>/` and writes your results to `/output/<clip_id>/`, with the same file names,
3. for the segmentation tracks, calls `reset()` at the start of every clip and `step(frame)` for every frame in order, timing each call,
4. for the generative track, calls `generate(masks)` once per clip of 16 masks.

The server scores the results with its own copies of `submission/run.py`, `atlas/metrics.py` and `evaluate.py`, so changing them has no effect. Put your changes in your track's folder or in new files.

> [!WARNING]
> Your code has **no internet access** on the server. Everything it needs must be in the zip: build your model with `pretrained=False` in `predictor.py`, and load your own checkpoint.

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

`data/eval/test` has the same layout and holds the data of the validation leaderboard.

## 3. Make the zip

From the `final-assignment` folder:

```bash
python -m submission.package --track consistency --checkpoint checkpoints/my-run/best.pt --base-image pytorch
```

This writes `submission_consistency.zip` with your track folder, every other course package your code imports (for example `segmentation/` when your consistency model reuses its U-Net), your weights and a `submission.yaml`. For a folder of weights, for example a Hugging Face model, pass the folder as `--checkpoint`.

Choose the base image that has what your code needs:

| `--base-image` | Contents |
| --- | --- |
| `pytorch` | PyTorch, torchvision, timm, NumPy, Pillow, SciPy, OpenCV |
| `transformers` | everything in `pytorch`, plus Hugging Face transformers and accelerate |
| `generative` | everything in `transformers`, plus diffusers |

Need a package that isn't in the image? Download its wheels for the image's Python (3.12, Linux) and add them with `--wheels`; the server installs them offline before your code runs:

```bash
pip download --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64 -d wheels <package>
python -m submission.package --track realtime --checkpoint best.pt --wheels wheels
```

## 4. Test it in the same environment (optional)

The base images are public, so you can run your submission exactly as the server does, for example on a few clips from `data/eval/val` copied to `local_data/`:

```bash
unzip submission_consistency.zip -d my_submission
docker run --rm --gpus all --network none \
  -v "$(pwd)/my_submission:/submission:ro" -v "$(pwd):/harness:ro" \
  -v "$(pwd)/local_data:/data:ro" -v "$(pwd)/local_output:/output" \
  -e PYTHONPATH=/submission:/harness \
  ghcr.io/tue-aria/nncv-pytorch:2027 \
  python /harness/submission/run.py --track consistency --input-dir /data --output-dir /output --checkpoint /submission/model.pt
```

Run it from the `final-assignment` folder. On Windows PowerShell, write `${PWD}` instead of `$(pwd)`. Without an NVIDIA GPU, leave out `--gpus all`: it is slow, but it shows whether your code runs. On Snellius, the same image works with Apptainer.

## 5. Upload it

Open the challenge website on the TU/e network or over VPN, go to the page of your track, choose your name, enter the password you received from the staff, and upload the zip.

- You can follow your submission in the queue; its scores and log appear on the submission page and under *My submissions*.
- **Validation**: scores on the `test` split, shown on the leaderboard. These are the scores you report in your paper.
- **Hidden test**: opens in the final week. Results are not shown; the ranking is announced on the challenge day, and the bonus points are based on it.
- The weekly number of successful submissions, the time limits and the upload size are listed on the website under *How to submit*. Failed submissions don't count.

> [!TIP]
> **Advanced: a complete Docker image.** If the base images and wheels are not enough, you can upload a complete image instead (`docker save -o image.tar <name>`). Build it with the [Dockerfile](../Dockerfile): it must contain Python with PyTorch, NumPy and Pillow, your code in `/app` and your weights in `/app/model.pt`. Images are large, so use this only if you need it.

Something doesn't work? See the [FAQ](faq.md#submission).

Next: [Report, pitch & grading](report.md).
