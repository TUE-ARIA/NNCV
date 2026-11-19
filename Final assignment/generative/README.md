# Generative track

Generate realistic surgical video from segmentation masks. Synthetic video like this can provide training data, simulation and teaching material.

- **Input:** 16 consecutive masks of 256×384 with train IDs 0-29, all at once
- **Output:** 16 RGB frames of 256×384
- **Ranked on:** FVD, lower is better, for submissions whose mask Dice is at least 70% of that of real frames
- **Baseline:** pix2pix, one frame at a time

## Scoring

| Metric | Measures | Better |
| --- | --- | --- |
| **FVD** (Fréchet Video Distance) | Realism of the clips, including motion: the distance between R3D-18 (Kinetics-400) features of real and generated clips. | lower |
| **FID** (Fréchet Inception Distance) | Realism of single frames: the distance between Inception-v3 (ImageNet) features of real and generated frames. | lower |
| **Mask Dice** | Whether the frames follow the masks. The [ATLAS model](https://github.com/rlpddejong/ATLAS-model), a strong segmenter trained on ATLAS-120k, segments your frames, and we compare its output with the input masks. The same is done for the real frames. | higher |

You are ranked on FVD, but only if your mask Dice is at least **70% of the mask Dice of the real frames**. This stops a model from producing realistic video that ignores the masks. FID is reported as well; discuss all three in your paper.

These numbers are only comparable within this challenge: papers use other feature extractors and more clips. The code is in [atlas/metrics.py](../atlas/metrics.py) and [atlas/reference.py](../atlas/reference.py).

## Baseline

[model.py](model.py) is **pix2pix**: a U-Net generator and a PatchGAN discriminator, trained with a least-squares GAN loss and an L1 loss ([train.py](train.py)). It generates every frame independently, so its videos flicker. FVD picks this up, and it is the first thing to improve.

## Your predictor

[predictor.py](predictor.py) defines the class the server calls:

| Method | Called | Does |
| --- | --- | --- |
| `__init__(checkpoint, device)` | once | loads your model |
| `generate(masks)` | once per clip | takes `(16, 256, 384)` uint8 masks, returns `(16, 256, 384, 3)` uint8 RGB frames |

You get the whole clip at once, so generation doesn't have to be causal.

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train generative
slurm/submit.sh evaluate generative checkpoints/generative-baseline/last.pt
```

The evaluation also runs the ATLAS reference segmenter, so it takes a few minutes on a GPU.

## Ideas

- **Temporal conditioning:** give the generator the previous generated frame, or use 3D convolutions or temporal attention.
- **A video discriminator** that judges short clips instead of single frames.
- **Stronger image models:** SPADE, OASIS, or conditional diffusion with ControlNet. Keep the server's time limit in mind.
- **Fine-tuning** a pretrained image or video diffusion model on ATLAS.

Next: [Submitting](../docs/submission.md).
