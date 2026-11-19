# Final Assignment: ATLAS Surgical Video Challenge

> [!TIP]
> This documentation is also a website, with search and navigation: **[tue-aria.github.io/NNCV](https://tue-aria.github.io/NNCV/)**.

In this assignment you train neural networks on **surgical video**. You segment anatomy and instruments in laparoscopic and robot-assisted surgery, make that segmentation stable over time or fast enough for the operating room, or generate realistic surgical video from segmentation masks.

The final assignment counts for **50% of your course grade**.

![Example frames from ATLAS-120k with their annotations](https://raw.githubusercontent.com/TimJaspers0801/ATLAS/main/figures/atlas120k.png)

*Frames from ATLAS-120k (rows 1 and 3) and their annotations (rows 2 and 4).*

## How to read this documentation

The pages build on each other, so read them in order. Each page ends with a link to the next.

| Step | Page | Contents |
| --- | --- | --- |
| 1 | **Overview** (this page) | The challenge, the rules and the timeline |
| 2 | [Setup](docs/setup.md) | Tools, accounts and your fork of the repository |
| 3 | [Snellius](docs/snellius.md) | Data, training and evaluation on the cluster |
| 4 | Tracks: [Dice](segmentation/README.md), [Consistency](consistency/README.md), [Real-time](realtime/README.md), [Generative](generative/README.md) | What each track measures and how its baseline works |
| 5 | [Submitting](docs/submission.md) | Packaging your model for the challenge server |
| 6 | [Report, pitch & grading](docs/report.md) | What you hand in, the challenge day, and how you are graded |
| 7 | [FAQ](docs/faq.md) | Solutions to common problems |

## The data: ATLAS-120k

[ATLAS-120k](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k) contains about 120,000 annotated frames from 100 surgical videos of 14 procedure types, grouped into about 500 short clips. Frames are sampled at 15 fps, and every annotated frame has a mask with **30 classes**: background, instruments, organs, vessels and other structures.

| Split | Use it for |
| --- | --- |
| `train` | Training. |
| `val` | Model selection and tuning. You may also train your final model on it. |
| `test` | Nothing: the validation server scores on it. These are the scores you report in your paper. |
| hidden test | New videos annotated by surgeons. Only for the final ranking, which is announced on the challenge day. |

The videos range from 4:3 (480×640) to 16:9 (1080×1920). Every frame is **center-cropped to the largest 3:2 region and resized to 256×384**, in training and on the server, for every track. The data is already on Snellius; you don't download anything.

## Tracks

Everyone takes part in the **Dice** track and chooses **at least one** of the other three. Each track has its own baseline, and you compare your method with the baseline of that track.

| Track | Question | Ranked on | Baseline |
| --- | --- | --- | --- |
| [Dice](segmentation/README.md) (mandatory) | How accurate is the segmentation of each frame? | Dice | U-Net with a ResNet-18 encoder |
| [Consistency](consistency/README.md) | Is the segmentation stable over time? | (Dice + Change Dice) / 2 | U-Net with ConvGRU memory |
| [Real-time](realtime/README.md) | What is the best accuracy at 25 frames per second or more? | Dice, if FPS ≥ 25 | LR-ASPP MobileNetV3 |
| [Generative](generative/README.md) | Can you generate realistic surgical video from masks? | FVD, if mask Dice ≥ 70% of real frames | pix2pix, frame by frame |

## Rules

- **Causal segmentation.** In the segmentation tracks, the prediction for frame *t* may only use frames up to *t*. The submission interface enforces this by giving your model one frame at a time.
- **Pretrained weights.** Any publicly available weights are allowed. State in your paper which ones you used and what they were trained on.
- **No test data.** Never train or tune on the `test` split.
- **Fixed server code.** The server uses its own copy of the scoring and submission code. Only your track's `predictor.py`, your model code and your weights change between submissions.
- **Cite your sources.** Build on the literature, but cite it. Plagiarism is not tolerated.

## Timeline

| When | What |
| --- | --- |
| TBD | Assignment starts; validation servers open. |
| TBD | Deadline for a Dice-track baseline submission, to check that your setup works. |
| TBD | Last day on Snellius and on the validation servers. |
| Final week (TBD) | Submit your final models to the hidden test servers. You don't get scores back. |
| TBD | Deadline for the paper and the code. |
| TBD | **Challenge day**: everyone pitches their work, and the final ranking on the hidden test set is announced. |

> [!IMPORTANT]
> You have no Snellius access in the final week. Train your final models, get their validation-server scores for your paper, and copy the checkpoints to your laptop before it starts.

## Your first week

- [ ] Install the tools and fork the repository ([Setup](docs/setup.md)).
- [ ] Set up Snellius and train the Dice baseline ([Snellius](docs/snellius.md)).
- [ ] Evaluate the baseline and submit it to the Dice validation server ([Submitting](docs/submission.md)).
- [ ] Read the track pages and choose your second track.

## Repository layout

```text
Final assignment/
├── atlas/             shared code: classes, datasets, metrics (as on the server), training helpers
├── segmentation/      Dice track:        model.py, train.py, predictor.py
├── consistency/       Consistency track: model.py, train.py, predictor.py
├── realtime/          Real-time track:   model.py, train.py, predictor.py
├── generative/        Generative track:  model.py, train.py, predictor.py
├── slurm/             cluster scripts: setup, submit, train and evaluate jobs
├── submission/run.py  entrypoint of the submission image (fixed)
├── evaluate.py        scores predictions with the server's metrics
├── tools/             builds the evaluation sets (already done for you)
├── docs/              the general documentation pages
└── Dockerfile         the submission image
```

Questions? Ask them in the **Discussions** section of this repository, so that everyone benefits from the answers.

Next: [Setup](docs/setup.md).
