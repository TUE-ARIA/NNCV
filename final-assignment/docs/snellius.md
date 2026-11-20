# Snellius

You train and evaluate your models on the Snellius cluster. You never run them directly: you **submit jobs** with Slurm, and every job runs inside a container that has PyTorch and the other libraries installed.

All cluster scripts are in the `slurm/` folder:

```text
slurm/
├── setup.sh             one-time setup: links the data and the containers
├── submit.sh            submits a job with the right partition, GPU and CPUs
├── servers/
│   └── snellius.env     cluster settings and shared paths (partition, containers, data)
└── jobs/
    ├── train.sh         train a model for a track
    └── evaluate.sh      run and score a checkpoint exactly like the challenge server
```

You only use `setup.sh` (once) and `submit.sh`. Run every command from the `final-assignment` folder.

## How to work

You can work in two ways, and both are fine:

- **Edit locally (recommended):** edit on your laptop, `git push`, then `git pull` on Snellius before you submit a job.
- **Edit on the cluster:** connect with VS Code **Remote - SSH**, edit on the server, and commit and push from there.

## Step 1: Clone your fork on Snellius

Log in to Snellius and clone your fork with your GitHub Personal Access Token (see [Setup](setup.md#tips)):

```bash
git clone https://<PAT>@github.com/<your-username>/NNCV.git
cd NNCV/final-assignment
```

Run `git pull` before every job, so the cluster runs your latest code.

## Step 2: One-time setup

```bash
bash slurm/setup.sh
```

This links the shared course data and the containers, and creates your `.env`. Afterwards you have:

- `data/atlas120k`: the dataset (`train`, `val`, `test`),
- `data/eval`: the `val` and `test` splits in the server's format (see [Submitting](submission.md)),
- `data/reference_segmenter.pth`: the ATLAS reference segmenter for the generative mask Dice,
- `containers/nncv-<image>.sif`: the three base images, the same ones the challenge server runs your submission in, so what works here works there.

These are links to a shared folder, so they don't use your storage quota. Don't write into them.

Every job runs in the `pytorch` container, unless you choose another one with `--image` (see [Step 4](#step-4-train)):

| Image | Contents |
| --- | --- |
| `pytorch` | PyTorch, torchvision, timm, NumPy, Pillow, SciPy, OpenCV, W&B |
| `transformers` | everything in `pytorch`, plus Hugging Face transformers and accelerate |
| `generative` | everything in `transformers`, plus diffusers |

If a container is missing from `containers/`, `bash slurm/setup.sh <image>` downloads it into its own file. This takes a few minutes.

## Step 3: Set your Weights & Biases key

`setup.sh` created `.env`, a copy of [.env.example](../.env.example) with the environment variables passed into the container. Open it (`nano .env`, or through VS Code or MobaXterm) and fill in your API key, which you find at [wandb.ai/authorize](https://wandb.ai/authorize):

```env
WANDB_API_KEY=<your W&B API key>
```

`.env` is in `.gitignore`, so your key never ends up in your public repository. Never add it with `git add -f`.

## Step 4: Train

```bash
slurm/submit.sh train segmentation                                   # the Dice baseline
slurm/submit.sh train consistency --epochs 30 --experiment-id gru-30ep
```

- The first argument after `train` is the track: `segmentation`, `consistency`, `realtime` or `generative`.
- Everything after it is passed to `python -m <track>.train`. Without flags, the defaults in that `train.py` are used. To see all options: `apptainer exec containers/nncv-pytorch.sif python -m segmentation.train --help`.
- Give every run its own `--experiment-id`. Checkpoints go to `checkpoints/<experiment-id>/`, so a reused name overwrites an earlier run.
- To use another container, put `--image` before the job name, for example `slurm/submit.sh --image transformers train segmentation`. Evaluate and submit with the same image you trained with.

## Step 5: Evaluate

```bash
slurm/submit.sh evaluate segmentation checkpoints/segmentation-baseline/best.pt
```

This runs your `predictor.py` on the `val` split exactly as the challenge server does, and prints the server's metrics at the end of the job log. Add `test` at the end to use the validation server's data instead.

## Following your jobs

`submit.sh` tells you where the log goes: `slurm/logs/<job>-<job id>.out`.

| Command | What it does |
| --- | --- |
| `squeue --me` | your queued and running jobs |
| `tail -f slurm/logs/train-<job id>.out` | follow a job's output |
| `scancel <job id>` | cancel a job |
| `accinfo` | your remaining compute budget |

Training curves and example predictions also appear in your Weights & Biases project `5lsm0-atlas`.

## Changing resources

Jobs ask for one A100 GPU with 18 CPU cores. Training may run for 4 hours, and evaluation for 1 hour. To change this for one job, put sbatch options before the job name:

```bash
slurm/submit.sh --time=08:00:00 train generative
```

Ask for enough time, but not much more: long jobs wait longer in the queue. The cluster settings themselves (partition, CPUs, container, data paths) are in [slurm/servers/snellius.env](../slurm/servers/snellius.env). You normally don't need to change them. See the [SURF partitions page](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting) for what each partition costs.

## Remember

- The last day on Snellius is **before the final week**. Have your final checkpoints trained and copied to your laptop by then.
- Don't run training or evaluation on the login node. Always use `slurm/submit.sh`.

Next: the tracks, starting with the [Dice track](../segmentation/README.md).
