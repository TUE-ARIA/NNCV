"""Export the ATLAS ViT-B Lightning checkpoint to the plain state dict used by `atlas/reference.py`.

    python -m tools.export_reference_weights ATLAS_vitb_val_iou_all=0.3498.ckpt data/reference_segmenter.pth

The checkpoint is on https://huggingface.co/rlpddejong/ATLAS-finetuned-weights.
"""
from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser("Export ATLAS reference weights")
    parser.add_argument("checkpoint", help="Lightning .ckpt file")
    parser.add_argument("output", help="Output .pth file")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    # keep the network only (drop the loss), without Lightning's "network." prefix
    weights = {k.removeprefix("network.").replace("._orig_mod", ""): v for k, v in state.items() if k.startswith("network.")}
    torch.save(weights, args.output)
    print(f"Saved {len(weights)} tensors to {args.output}")


if __name__ == "__main__":
    main()
