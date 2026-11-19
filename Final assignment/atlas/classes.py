"""ATLAS-120k class definitions.

The raw masks (`machine_masks/*.png`) store original class IDs (0-46).
Training and evaluation use 30 consolidated train IDs (0-29), where 0 is background.
Use `to_train_ids` to convert a raw mask.
"""
import numpy as np
import torch

NUM_CLASSES = 30
BACKGROUND = 0

# (name, RGB colour) per train ID.
TRAIN_CLASSES = [
    ("Background", (0, 0, 0)),
    ("Tools/camera", (255, 255, 255)),
    ("Vein", (0, 0, 255)),
    ("Artery", (255, 0, 0)),
    ("Nerve", (255, 255, 0)),
    ("Small intestine", (0, 255, 0)),
    ("Colon/rectum", (0, 200, 100)),
    ("Abdominal wall", (200, 150, 100)),
    ("Diaphragm", (250, 150, 100)),
    ("Fat", (255, 200, 100)),
    ("Liver", (150, 100, 50)),
    ("Bile/lymph duct", (0, 255, 255)),
    ("Gallbladder", (0, 200, 255)),
    ("Hepatic ligament", (255, 150, 50)),
    ("Cystic plate", (255, 220, 200)),
    ("Stomach", (200, 100, 200)),
    ("Spleen", (200, 0, 200)),
    ("Uterus", (255, 0, 150)),
    ("Ovary", (255, 100, 200)),
    ("Oviduct", (200, 100, 255)),
    ("Prostate", (150, 0, 100)),
    ("Urethra", (255, 200, 255)),
    ("Ligated plexus", (150, 100, 75)),
    ("Seminal vesicles", (200, 0, 150)),
    ("Non anatomical", (50, 50, 50)),
    ("Bladder", (255, 150, 255)),
    ("Lung", (150, 200, 255)),
    ("Airway (bronchus/trachea)", (0, 150, 255)),
    ("Esophagus", (255, 100, 100)),
    ("Pericardium", (200, 200, 255)),
]

# Original class ID (index) -> train ID. Follows the official ATLAS-120k mapping.
_RAW_TO_TRAIN = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,         # 0-9
    3, 2, 10, 11, 12, 2, 13, 14, 15, 11,  # 10-19
    9, 11, 16, 17, 18, 19, 20, 21, 22, 23,  # 20-29
    24, 25, 0, 26, 27, 28, 29, 2, 11, 4,  # 30-39
    0, 24, 0, 0, 0, 0, 0,                 # 40-46
]
RAW_TO_TRAIN_LUT = np.zeros(256, dtype=np.uint8)
RAW_TO_TRAIN_LUT[: len(_RAW_TO_TRAIN)] = _RAW_TO_TRAIN

PALETTE = np.array([color for _, color in TRAIN_CLASSES], dtype=np.uint8)


def to_train_ids(raw_mask: np.ndarray) -> np.ndarray:
    """Map a raw ATLAS mask (H, W) with original IDs to train IDs."""
    return RAW_TO_TRAIN_LUT[raw_mask]


def colorize(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Turn a train-ID mask (..., H, W) into an RGB image (..., H, W, 3) for visualisation."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return PALETTE[mask.astype(np.int64)]
