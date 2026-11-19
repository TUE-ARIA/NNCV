# Final Assignment: ATLAS Surgical Video Challenge

> 📖 Read this documentation as a website, with search and navigation, at **[tue-aria.github.io/NNCV](https://tue-aria.github.io/NNCV/)**.

In this assignment you train neural networks on **surgical videos**. You segment anatomy and instruments in laparoscopic and robotic surgery, make those segmentations stable over time or fast enough for the operating room, or generate realistic surgical video from segmentation masks. The final assignment is **50% of your course grade**.

## How to read this documentation

Read the pages in this order. Each page ends with a link to the next one.

1. **Overview** (this page): the challenge, the rules and the timeline.
2. [**Setup**](docs/setup.md): tools, accounts, and your fork of the repository.
3. [**Snellius**](docs/snellius.md): get the data, train a baseline, evaluate it.
4. **Tracks**: what each benchmark measures and how its baseline works: [Dice](segmentation/README.md), [Consistency](consistency/README.md), [Real-time](realtime/README.md), [Generative](generative/README.md).
5. [**Submitting**](docs/submission.md): package your model and submit it to the server.
6. [**Report & grading**](docs/report.md): what you hand in and how it is graded.
7. [**FAQ**](docs/faq.md): answers to common problems.

## The data: ATLAS-120k

[ATLAS-120k](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k) has about 120,000 annotated frames from 100 surgical videos (14 procedure types), grouped into ~500 short clips. Frames are sampled at 15 fps, and every annotated frame has a mask with **30 classes**: background, tools, organs, vessels and more (see `atlas/classes.py`).

| Split | Use it for |
|---|---|
| `train` | Training. |
| `val` | Model selection and tuning. You may also train your final model on it. |
| `test` | **Do not train on it.** It is the validation set of the challenge server. |
| hidden test | New videos annotated by surgeons. Only used in the final week (see [Timeline](#timeline)). |

The videos range from 4:3 (480×640) to 16:9 (1080×1920). To keep shapes consistent without stretching, **every frame is center-cropped to the largest 3:2 region and resized to 256×384**, in training and on the server, for every track.

You don't download anything: the data is in a shared folder on Snellius (see [Snellius](docs/snellius.md)).

## Benchmarks

**Everyone** submits to the **Dice** track, and picks **at least one** of the other three. Every track has its own baseline. In your report, compare your method to the baseline **of that track**.

| Track | Question | Ranked on | Baseline |
|---|---|---|---|
| [**Dice**](segmentation/README.md) (mandatory) | How accurate is the segmentation per frame? | Dice | U-Net, ResNet-18 encoder |
| [**Consistency**](consistency/README.md) | Is the segmentation stable over time, without flicker? | (Dice + Change Dice) / 2 | U-Net + ConvGRU memory |
| [**Real-time**](realtime/README.md) | What is the best accuracy at ≥ 25 frames per second? | Dice, if FPS ≥ 25 | LR-ASPP MobileNetV3 |
| [**Generative**](generative/README.md) | Can you generate realistic surgical video from masks? | FVD, if mask Dice ≥ 70% of real frames | pix2pix, frame by frame |

### Rules

- **Causal segmentation.** In all segmentation tracks, the prediction for frame *t* may only use frames up to *t*. The submission interface enforces this: your model receives the frames one at a time.
- **Pretrained weights.** Any publicly available weights are allowed. Say in your report which ones you used and what they were trained on. Some surgical models may have been trained on the same YouTube videos as ATLAS, so discuss that if it applies.
- **No test data in training.** Never train or tune on the `test` split.
- **The server code is fixed.** Between submissions, only your track's `predictor.py`, your model code and your weights change.

## Timeline

| When | What |
|---|---|
| TBD | Assignment starts, servers open. |
| TBD | Deadline for a Dice-track baseline submission (proves your setup works). |
| TBD | Last day on Snellius and on the validation servers. |
| Final week (TBD) | Hidden test servers open: TBD submissions per track. Report the scores you get back in your paper. No Snellius access in this week, so have your final models trained before it starts. |
| TBD | Report and code deadline. |

## Your first week

- [ ] Install the tools and fork the repository ([Setup](docs/setup.md)).
- [ ] Link the data and start a Dice-track baseline run on Snellius ([Snellius](docs/snellius.md)).
- [ ] Evaluate the baseline locally, then submit it to the Dice validation server ([Submitting](docs/submission.md)).
- [ ] Read the track pages and choose your second track.

## Repository layout

```text
Final assignment/
├── atlas/             shared code: classes, datasets, metrics (same as the server), training helpers
├── segmentation/      Dice track:        model.py, train.py, predictor.py, README.md
├── consistency/       Consistency track: model.py, train.py, predictor.py, README.md
├── realtime/          Real-time track:   model.py, train.py, predictor.py, README.md
├── generative/        Generative track:  model.py, train.py, predictor.py, README.md
├── submission/run.py  entrypoint of the submission image (do not change)
├── evaluate.py        scores predictions locally with the server's metrics
├── tools/             builds the server-style evaluation sets (already done for you)
├── docs/              the general documentation pages
├── main.sh, jobscript_slurm.sh, download_data.sh   Snellius scripts
└── Dockerfile         submission image
```

Run all Python commands from the `Final assignment` folder with `python -m`, e.g. `python -m segmentation.train`.

Use ideas from the literature, but **cite all sources**: plagiarism is not tolerated. Ask your questions in the **Discussions** section of this repository, so that everyone benefits from the answers. Good luck! 🚀

Next: [Setup](docs/setup.md).
