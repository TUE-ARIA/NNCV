"""Train the real-time baseline. Uses the same training loop as the Dice track.

Run from the `Final assignment` folder:
    python -m realtime.train --data-dir ./data/atlas120k --experiment-id my-run
"""
from realtime.model import Model
from segmentation.train import get_args_parser, main

if __name__ == "__main__":
    parser = get_args_parser("realtime")
    parser.set_defaults(batch_size=64, lr=1e-3)
    main(parser.parse_args(), Model(pretrained=True), track="realtime")
