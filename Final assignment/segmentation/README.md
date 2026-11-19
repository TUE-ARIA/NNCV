# Dice track

Segment every frame as accurately as possible. This track is **mandatory** for everyone.

- **Input:** One RGB frame of 256×384 at a time, in temporal order
- **Output:** A 256×384 mask with train IDs 0-29
- **Ranked on:** Dice, higher is better
- **Baseline:** U-Net with an ImageNet-pretrained ResNet-18 encoder

## Scoring

For each frame, we compute the Dice score of every foreground class (train IDs 1-29) that appears in the ground truth **or** in your prediction:

$$\text{Dice}_c = \frac{2\,|P_c \cap G_c|}{|P_c| + |G_c|}$$

and average over those classes. Frame scores are averaged per clip, and clip scores over all clips, so every clip counts equally.

- A class that appears in neither mask is skipped.
- A class you predict that is not in the ground truth scores 0, so false positives count.
- Background is not scored itself, but predicting a class where there is background creates false positives.

The code is `frame_dice` in [atlas/metrics.py](../atlas/metrics.py).

## Baseline

[model.py](model.py) is a U-Net with an ImageNet-pretrained **ResNet-18** encoder. [train.py](train.py) trains it with cross-entropy, AdamW, a cosine learning-rate schedule and bfloat16 mixed precision, and keeps the checkpoint with the best validation Dice in `checkpoints/<experiment-id>/best.pt`.

## Your predictor

[predictor.py](predictor.py) defines the class the server calls. Change what happens inside, but keep the class name and the methods:

| Method | Called | Does |
| --- | --- | --- |
| `__init__(checkpoint, device)` | once | loads your model |
| `reset()` | at the start of every clip | clears any memory (the baseline has none) |
| `step(frame)` | for every frame, in order | takes a `(256, 384, 3)` uint8 RGB frame, returns a `(256, 384)` uint8 mask |

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train segmentation
slurm/submit.sh evaluate segmentation checkpoints/segmentation-baseline/best.pt
```

## Ideas

- **Class imbalance:** thin structures such as ducts, vessels and nerves are rare. Try a Dice or focal loss, or class-balanced sampling.
- **Stronger models:** other encoders (ConvNeXt, Swin, DINOv2/v3) or segmentation heads (DeepLabV3+, SegFormer, Mask2Former).
- **Augmentation** that fits surgery: colour jitter, blur, smoke and specular highlights.
- **Test-time augmentation**, such as averaging over flips.

Next: [Consistency track](../consistency/README.md).
