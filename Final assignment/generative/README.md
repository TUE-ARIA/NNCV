# Generative track

**Goal:** generate realistic surgical video from segmentation masks. Given 16 consecutive masks, produce the 16 matching RGB frames. Synthetic video like this can be used to build training data, simulators or education material.

## Input and output

- Input: 16 consecutive masks of **256×384** (height × width) with train IDs.
- Output: 16 RGB frames of 256×384.
- As in all tracks, every frame is center-cropped to the largest 3:2 region and resized to 256×384.
- You get the whole clip at once, so the generation does not have to be causal.

## Scoring
Ranked on **FVD** (lower is better), but only if your frames follow the masks: your **mask Dice must be at least 70% of the mask Dice of the real frames**. FID is reported as well. Discuss all three in your report.

| Metric | What it measures | Lower/higher is better |
|---|---|---|
| **FVD** (Fréchet Video Distance) | Realism of the *clips*, including motion. Distance between R3D-18 (Kinetics-400) features of real and generated clips. | lower |
| **FID** (Fréchet Inception Distance) | Realism of single *frames*. Distance between Inception-v3 (ImageNet) features of real and generated frames. | lower |
| **Mask Dice** | Does the frame follow the mask? The [ATLAS model](https://github.com/rlpddejong/ATLAS-model) (ViT-B), a strong segmenter trained on ATLAS-120k, segments your frames, and its output is compared to the input masks. The same is done for the real frames, which gives the reference value for the 70% floor. | higher |

These scores are only comparable within this challenge: the feature extractors and number of clips differ from those used in papers. See [atlas/metrics.py](../atlas/metrics.py) for the code.

## Baseline
[model.py](model.py) is **pix2pix**: a U-Net generator and a PatchGAN discriminator, trained with an LSGAN loss plus an L1 loss ([train.py](train.py)). It generates every frame independently, so the result flickers over time. That is what FVD picks up and what you can improve.

## Interface
Your [predictor.py](predictor.py) must define `Predictor` with:

- `__init__(checkpoint, device)`: load your model.
- `generate(masks)`: takes `(16, 256, 384)` uint8 masks and returns `(16, 256, 384, 3)` uint8 RGB frames.

## Run it
```bash
sbatch jobscript_slurm.sh generative
python -m submission.run --track generative --checkpoint checkpoints/generative-baseline/last.pt \
    --input-dir data/eval/val/generative/masks --output-dir predictions/generative
python evaluate.py generative --pred-dir predictions/generative --gt-dir data/eval/val/generative \
    --reference-model data/reference_segmenter.pth
```
FID and FVD need many samples to be reliable, and they run faster on a GPU, so evaluate in a Slurm job.

## Ideas

- Temporal conditioning: give the generator the previous generated frame, or use 3D convolutions or temporal attention.
- Better image models: SPADE, OASIS, or conditional diffusion with ControlNet. Keep the server time limit in mind.
- A video discriminator that judges short clips instead of single frames.
- Fine-tune a pretrained image or video diffusion model on ATLAS.

Next: [Submitting](../docs/submission.md).
