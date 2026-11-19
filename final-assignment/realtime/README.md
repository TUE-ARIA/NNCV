# Real-time track

Get the most accurate segmentation that is fast enough to show live during surgery.

- **Input:** One RGB frame of 256×384 at a time, in temporal order
- **Output:** A 256×384 mask with train IDs 0-29
- **Ranked on:** Dice, for submissions that reach at least 25 FPS on the server
- **Baseline:** LR-ASPP with an ImageNet-pretrained MobileNetV3-Large

## Scoring

- **FPS:** the server times every `step(frame)` call on its GPU, preprocessing and postprocessing included, with one frame at a time. The first 10 frames are a warm-up and are not timed. FPS is the number of timed frames divided by their total time.
- **Dice:** the same metric as in the [Dice track](../segmentation/README.md). Submissions with FPS ≥ 25 are ranked on it. Slower submissions are listed, but not ranked.

> [!NOTE]
> FPS depends on the GPU. Your local FPS (in `predictions/realtime-val/timing.json` after an evaluate job) is only an indication; the server's number counts.

## Baseline

[model.py](model.py) is torchvision's **LR-ASPP** segmentation head on a **MobileNetV3-Large** backbone. [train.py](train.py) uses the same training loop as the Dice track.

## Your predictor

The same interface as the [Dice track](../segmentation/README.md#your-predictor): `reset()` at the start of every clip, then `step(frame)` for every frame. Frames arrive one at a time, so you can reuse work from the previous frame, but you can't batch frames.

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train realtime
slurm/submit.sh evaluate realtime checkpoints/realtime-baseline/best.pt
```

## Ideas

- **Profile first:** find out how much time goes to preprocessing, the model, upsampling and copying to the CPU.
- **Faster inference:** half precision, `torch.compile`, channels-last memory layout, or TensorRT (it must run inside the submission image).
- **Distillation** from a large Dice-track model into a small one.
- **Use time:** run a heavy model only every *k* frames and propagate its result in between.
- Plot Dice against FPS for several settings, and show where your models lie.

Next: [Generative track](../generative/README.md).
