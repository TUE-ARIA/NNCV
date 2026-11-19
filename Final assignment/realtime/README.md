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
```bash
sbatch jobscript_slurm.sh realtime
python -m submission.run --track realtime --checkpoint checkpoints/realtime-baseline/best.pt \
    --input-dir data/eval/val/segmentation/images --output-dir predictions/realtime
python evaluate.py realtime --pred-dir predictions/realtime --gt-dir data/eval/val/segmentation/labels
```

## Ideas

- Find where the time goes: preprocessing, the model, upsampling, or copying to the CPU.
- Half precision, `torch.compile`, channels-last memory layout, or TensorRT (it must run inside the submission image).
- Knowledge distillation from a large Dice-track model into a small one.
- Lower input resolution with a better upsampling head. Plot Dice against FPS for several settings.
- Use time: run the heavy model only every *k* frames and propagate the result in between.

Next: [Generative track](../generative/README.md).
