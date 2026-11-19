# Consistency track

Segment accurately **and** stably over time. A mask that flickers between frames distracts a surgeon, even when every single frame looks fine.

- **Input:** One RGB frame of 256×384 at a time, in temporal order
- **Output:** A 256×384 mask with train IDs 0-29
- **Ranked on:** Consistency score = (Dice + Change Dice) / 2, higher is better
- **Baseline:** The Dice-track U-Net with a ConvGRU memory

## Scoring

- **Dice** is the same metric as in the [Dice track](../segmentation/README.md).
- **Change Dice** measures whether your mask changes where the ground truth changes. For each pair of consecutive annotated frames, we mark the pixels whose label changes, in your prediction and in the ground truth. Change Dice is the Dice between these two change maps, summed over all pairs of a clip and averaged over clips.

Change Dice punishes two kinds of error:

- **flicker**: your label changes where the ground truth doesn't,
- **lag**: your mask doesn't follow real motion. Predicting the same mask for every frame therefore doesn't help.

The code is `ClipScorer` in [atlas/metrics.py](../atlas/metrics.py).

## Baseline

[model.py](model.py) adds a **ConvGRU** at the bottleneck of the Dice-track U-Net. Its hidden state carries information from earlier frames to the current one. [train.py](train.py) trains on windows of 8 consecutive frames with per-frame cross-entropy, backpropagating through the whole window, and keeps the checkpoint with the best validation consistency score.

> [!TIP]
> Start from a trained Dice-track model: `slurm/submit.sh train consistency --init-from checkpoints/segmentation-baseline/best.pt`.

## Your predictor

The same interface as the [Dice track](../segmentation/README.md#your-predictor): `reset()` at the start of every clip, then `step(frame)` for every frame. Frames arrive in order and you can't see future frames, so keep the memory you need (hidden state, previous masks, features) in `self` and clear it in `reset()`.

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train consistency
slurm/submit.sh evaluate consistency checkpoints/consistency-baseline/best.pt
```

## Ideas

- **Temporal losses** that penalise changes between neighbouring predictions that the labels don't have.
- **Propagating features** with optical flow, or attention over a memory of past frames, as in XMem or SAM 2.
- **Causal post-processing**, such as exponential smoothing of the class probabilities.
- Report what your method gains in Change Dice against what it costs in Dice.

Next: [Real-time track](../realtime/README.md).
