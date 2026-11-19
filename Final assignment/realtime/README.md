# Real-time track

**Goal:** the most accurate segmentation that runs at **at least 25 frames per second**, fast enough to show live during surgery.

## Scoring

- **FPS**: [submission/run.py](../submission/run.py) times every `step(frame)` call on the server GPU with batch size 1, preprocessing and postprocessing included. The first 10 frames are not timed (warm-up). FPS = timed frames / total time.
- Submissions with **FPS ≥ 25** are ranked on **Dice** (same metric as the [Dice track](../segmentation/README.md)). Slower submissions are listed but not ranked.

Your local FPS (in `predictions/<track>/timing.json`) depends on your GPU. Only the server's number counts.

## Baseline
[model.py](model.py) is torchvision's **LR-ASPP** head on an ImageNet-pretrained **MobileNetV3-Large**. It trains with the same loop as the Dice track ([train.py](train.py) reuses [segmentation/train.py](../segmentation/train.py)). The [predictor](predictor.py) resizes on the GPU, which matters for speed with full-HD input frames.

## Interface
Same as the Dice track: `Predictor` with `reset()` and `step(frame)`. Frames come one at a time and in order, so you can reuse work from the previous frame, but you can't batch future frames.

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train realtime
slurm/submit.sh evaluate realtime checkpoints/realtime-baseline/best.pt
```

## Ideas

- Find where the time goes: preprocessing, the model, upsampling, or copying to the CPU.
- Half precision, `torch.compile`, channels-last memory layout, or TensorRT (it must run inside the submission image).
- Knowledge distillation from a large Dice-track model into a small one.
- Lower input resolution with a better upsampling head. Plot Dice against FPS for several settings.
- Use time: run the heavy model only every *k* frames and propagate the result in between.

Next: [Generative track](../generative/README.md).
