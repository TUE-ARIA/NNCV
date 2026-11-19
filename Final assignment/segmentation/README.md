# Dice track (mandatory)

**Goal:** segment every frame as accurately as possible.

## Scoring
For every frame, we compute the Dice score of each foreground class (train IDs 1-29) that appears in the ground truth **or** in your prediction, and average those per-class scores:

$$\text{Dice}_c = \frac{2\,|P_c \cap G_c|}{|P_c| + |G_c|}$$

- A class that appears in neither mask is skipped. A class you predict but that is not in the ground truth scores 0, so false positives count.
- Background is not scored directly. It still matters, because labelling background as a class creates false positives.
- Frame scores are averaged per clip, then over clips, so every clip counts equally.
- Frames and masks are 256×384: center-cropped to 3:2 and resized, like in training.

The code is `frame_dice` in [atlas/metrics.py](../atlas/metrics.py).

## Baseline
[model.py](model.py) is a U-Net with an ImageNet-pretrained **ResNet-18** encoder. [train.py](train.py) trains it with cross-entropy at 256×384, using AdamW, a cosine learning-rate schedule and bfloat16 mixed precision. It saves the checkpoint with the best validation Dice to `checkpoints/<experiment-id>/best.pt`.

## Interface
Your [predictor.py](predictor.py) must define `Predictor` with:

- `__init__(checkpoint, device)`: load your model.
- `reset()`: called at the start of every clip.
- `step(frame)`: takes one `(H, W, 3)` uint8 RGB frame and returns an `(H, W)` uint8 mask with train IDs.

## Run it
```bash
sbatch jobscript_slurm.sh segmentation                       # train on Snellius
python -m submission.run --track segmentation --checkpoint checkpoints/segmentation-baseline/best.pt \
    --input-dir data/eval/val/segmentation/images --output-dir predictions/segmentation
python evaluate.py segmentation --pred-dir predictions/segmentation --gt-dir data/eval/val/segmentation/labels
```

## Ideas

- Class imbalance: thin structures such as ducts, vessels and nerves are rare. Try Dice or focal loss, or class-balanced sampling.
- Stronger encoders (ConvNeXt, Swin, DINOv2/v3) or segmentation heads (DeepLabV3+, SegFormer, Mask2Former).
- Augmentation that fits surgery: colour jitter, blur, smoke and specular highlights.
- Higher resolution, or test-time augmentation.

Next: [Consistency track](../consistency/README.md).
