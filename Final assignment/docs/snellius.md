# Snellius

You train your models on the Snellius HPC cluster with Slurm. All jobs run inside an Apptainer container that has PyTorch and the other libraries installed.

You can work in two ways, and both are fine:

- **Edit locally (recommended):** edit on your laptop, `git push`, then `git pull` on Snellius before you submit a job.
- **Edit on the cluster:** connect with VSCode **Remote-SSH**, edit on the server, and commit and push from there.

## Step 1: Clone your fork on Snellius
Create a GitHub Personal Access Token (`Settings` → `Developer settings` → `Personal access tokens` → `Tokens (classic)`), log in to Snellius and run:

```bash
git clone https://<PAT>@github.com/<your-username>/NNCV.git
cd "NNCV/Final assignment"
```

Run `git pull` before every job so the cluster runs your latest code.

## Step 2: Link the data and get the container (once)

```bash
sbatch download_data.sh
```

When the job has finished you should have:

- `container.sif`: the training container,
- `data/atlas120k`: the dataset (`train`, `val`, `test`),
- `data/eval`: the `val` and `test` splits in the server's format (see [Submitting](submission.md)),
- `data/reference_segmenter.pth`: the ATLAS reference segmenter that computes mask Dice in the generative track.

These are links to a shared course folder, so they don't use your storage quota. Don't write into them.

## Step 3: Set your environment variables
[.env](../.env) holds the environment variables passed into the container. Open it (`nano .env`, or through VSCode or MobaXterm) and set at least:

```env
WANDB_API_KEY=<your W&B API key>
WANDB_DIR=/home/<username>/wandb
```

Never commit your API key.

## Step 4: Submit a training job
Give the track as an argument:

```bash
sbatch jobscript_slurm.sh segmentation     # or: consistency, realtime, generative
```

[jobscript_slurm.sh](../jobscript_slurm.sh) asks for the hardware and starts [main.sh](../main.sh) inside the container. `main.sh` holds the training command and hyperparameters for each track, so edit those there. Give every run its own `--experiment-id`, or it overwrites the checkpoints of an earlier run in `checkpoints/<experiment-id>/`.

To score a checkpoint on the `val` split exactly as the server would:

```bash
sbatch jobscript_slurm.sh evaluate segmentation checkpoints/segmentation-baseline/best.pt
```

## Slurm parameters

| Parameter | Meaning |
|---|---|
| `--nodes=1`, `--ntasks=1` | one process on one machine |
| `--cpus-per-task=18` | CPU cores, mostly used for data loading (`--num-workers`) |
| `--gpus=1` | one GPU |
| `--partition=gpu_a100` | the A100 partition. See the [SURF partitions page](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting) for what each partition costs. |
| `--time=04:00:00` | the job is stopped after this time. Ask for enough time, but not much more: long jobs wait longer in the queue. |

## Useful commands

- `squeue -u $USER`: your queued and running jobs.
- `scancel <job_id>`: cancel a job.
- `tail -f slurm-<job_id>.out`: follow the output of a job.
- `accinfo`: your remaining compute budget.

## Remember

- The last day on Snellius is **before the final week**. Have your final checkpoints trained and copied to your laptop by then.
- Don't run training or evaluation on the login node. Always use `sbatch`.

Next: the tracks, starting with the [Dice track](../segmentation/README.md).
