# FAQ & troubleshooting

Can't find your problem here? Search the **Discussions** section of the repository, or ask your question there.

## General

### Which tracks do I have to do?

The Dice track, plus at least one of Consistency, Real-time and Generative. See the [overview](../README.md#tracks).

### May I use pretrained models?

Yes, any publicly available weights. Say in your report which ones you used and what they were trained on.

### May I train on the `val` split? On `test`?

On `val`, yes, but then you need another way to select your model. On `test`, never: it is the validation set of the challenge server.

### May I use future frames in the segmentation tracks?

No. Your `Predictor` gets one frame at a time and can only remember the past.

## Snellius

### `No clips found in data/atlas120k/...`

The data links are missing. Run `bash slurm/setup.sh` once from the `final-assignment` folder, and check `ls -l data`.

### My job stopped before training finished.

It hit its time limit. Ask for more time (`slurm/submit.sh --time=08:00:00 train ...`), use fewer epochs, or use a larger `--frame-stride`. Training saves `best.pt` after every epoch that improves validation, so you don't lose everything.

### `CUDA out of memory`

Lower the batch size, e.g. `slurm/submit.sh train consistency --batch-size 4`. For the consistency track, `--clip-length` matters too.

### Training is slow, but the GPU is barely used.

Data loading is the bottleneck. Keep `--num-workers` close to the 18 CPU cores of a job, and use `--frame-stride` to skip near-identical neighbouring frames.

### W&B doesn't show my run.

Check `WANDB_API_KEY` in `.env`, and look in `slurm/logs/train-<job id>.out` for W&B messages.

## Submission

### `Missing key(s)` or `Unexpected key(s)` when loading `model.pt`.

The checkpoint doesn't match the model that `predictor.py` builds, for example when you trained with other settings or submitted a checkpoint from another track.

### My code tries to download weights and fails.

The server has no internet. Build the model with `pretrained=False` in `predictor.py` and load your own checkpoint.

### `mask shape ... must equal frame shape ...`

Your mask must have the same size as the input frame, 256×384. If your model works at another resolution, resize its logits back before taking the argmax (see `logits_to_mask` in `atlas/inference.py`).

### `expected uint8 frames of shape (16, 256, 384, 3)`

A generative `Predictor` must return one uint8 RGB frame per input mask, at 256×384.

### `No module named ...`

The package is not in your base image. Choose a larger one (`--base-image transformers` or `generative`), or add its wheels with `--wheels` (see [Submitting](submission.md#3-make-the-zip)). If the missing module is one of your own folders, make sure your track's code imports it, so `submission.package` includes it.

### The scores look wrong, or are 0.

Check that you made the zip for the right `--track` and with the right checkpoint, and test it locally first (see [Submitting](submission.md#4-test-it-in-the-same-environment-optional)).

### My zip is very large.

It should hold only code and weights. If it holds more, check that `--checkpoint` points to one file (or to a folder with only the weights).

### My local FPS is different from the server's.

FPS depends on the GPU. Only the server's number counts for the real-time track.
