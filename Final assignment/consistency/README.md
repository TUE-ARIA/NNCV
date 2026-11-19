# Consistency track

**Goal:** a segmentation that is accurate *and* stable over time. Surgeons are distracted by masks that flicker between frames, even when every single frame looks fine.

## Scoring
Ranked on the **consistency score = (Dice + Change Dice) / 2**.

- **Dice**: the same metric as in the [Dice track](../segmentation/README.md).
- **Change Dice**: for every pair of neighbouring annotated frames (directly consecutive at 15 fps), we mark the pixels whose label **changes**, both in your prediction and in the ground truth. Change Dice is the Dice between these two change maps, summed over all pairs of a clip and then averaged over clips.
  - *Flicker* (your label changes where the ground truth doesn't) lowers it.
  - *Lag* (your mask doesn't follow real motion) lowers it too, so predicting the same mask on every frame does not help.

The code is `ClipScorer` in [atlas/metrics.py](../atlas/metrics.py).

## Baseline
[model.py](model.py) is the Dice-track U-Net with a **ConvGRU** at the bottleneck. The GRU state carries information from earlier frames to the current one. [train.py](train.py) trains on windows of 8 consecutive frames with per-frame cross-entropy, backpropagating through the whole window. It keeps the checkpoint with the best validation consistency score.

Tip: initialise the U-Net from your Dice-track model with `--init-from checkpoints/<run>/best.pt`.

## Interface
Same as the Dice track: `Predictor` with `reset()` and `step(frame)`. Frames come in temporal order, so keep your memory (hidden state, previous masks, features) in `self` and clear it in `reset()`. You can't look ahead at future frames.

## Run it

On Snellius (see [Snellius](../docs/snellius.md)):

```bash
slurm/submit.sh train consistency
slurm/submit.sh evaluate consistency checkpoints/consistency-baseline/best.pt
```

## Ideas

- Temporal losses: penalise changes between neighbouring predictions that the labels don't have.
- Feature propagation with optical flow, or attention over a memory of past frames (as in XMem or SAM2).
- Causal post-processing: exponential smoothing of the probabilities, or flow-warped averaging.
- Compare what your method gains in Change Dice with what it costs in Dice.

Next: [Real-time track](../realtime/README.md).
