"""Dataset loading for the MRL eye-state classifier.

The data lives under ``data/<split>/<class>/`` where ``split`` is one of
``train`` / ``val`` / ``test`` and ``class`` is ``awake`` or ``sleepy`` (the MRL
fork's labels for open- vs closed-eye infrared crops). Images are variable-sized,
single-channel grayscale PNGs; the transforms below normalize them to the fixed
``(1, 32, 32)`` float tensors that ``EyeStateCNN`` expects.

Label convention (kept consistent across the pipeline): ``0 = open`` (awake),
``1 = closed`` (sleepy). ``ImageFolder`` assigns indices by sorting class names, so
``awake`` -> 0 and ``sleepy`` -> 1 fall out automatically; ``build_dataset`` asserts
this so the mapping can never silently flip.
"""

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Fixed pipeline constants -------------------------------------------------

IMG_SIZE = 32
CLASS_TO_IDX = {"awake": 0, "sleepy": 1}  # 0 = open, 1 = closed

# Normalization stats computed over a 3,000-image sample of the training split
# (grayscale, resized to 32x32, scaled to [0, 1]).
MEAN = (0.3454,)
STD = (0.1319,)

# data/ sits next to src/ at the project root.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DEFAULT_NUM_WORKERS = 4


def build_transform(train: bool) -> transforms.Compose:
    """Build the preprocessing pipeline for a split.

    ``train=True`` adds light augmentation (small rotation + horizontal flip);
    eye open/closed state is invariant to both, and the flip effectively shares
    left/right-eye crops. Val/test use the deterministic pipeline only.
    """
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if train:
        ops += [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),  # -> float tensor in [0, 1], shape (1, 32, 32)
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(ops)


def build_dataset(split: str, data_root: os.PathLike | str = DATA_ROOT,
                  augment: bool | None = None) -> datasets.ImageFolder:
    """Create an ``ImageFolder`` for ``split`` ("train" / "val" / "test").

    Augmentation defaults to on for the train split and off otherwise; pass
    ``augment`` explicitly to override.
    """
    if augment is None:
        augment = split == "train"
    root = Path(data_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {root}")

    ds = datasets.ImageFolder(str(root), transform=build_transform(train=augment))
    if ds.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Unexpected class-to-index mapping for {root}: {ds.class_to_idx}. "
            f"Expected {CLASS_TO_IDX} (0=open/awake, 1=closed/sleepy)."
        )
    return ds


def build_dataloader(split: str, batch_size: int = 64, *, shuffle: bool | None = None,
                     num_workers: int = DEFAULT_NUM_WORKERS,
                     data_root: os.PathLike | str = DATA_ROOT,
                     augment: bool | None = None) -> DataLoader:
    """Create a ``DataLoader`` for ``split``.

    Shuffling defaults to on for the train split and off otherwise. On macOS the
    default multiprocessing start method is ``spawn``, so callers using
    ``num_workers > 0`` must create loaders under an ``if __name__ == "__main__"``
    guard (as ``train.py`` does).
    """
    if shuffle is None:
        shuffle = split == "train"
    ds = build_dataset(split, data_root=data_root, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


if __name__ == "__main__":
    # Smoke test: report sizes and one batch's shape/label range per split.
    for _split in ("train", "val", "test"):
        _ds = build_dataset(_split)
        print(f"{_split:5s}  n={len(_ds):6d}  class_to_idx={_ds.class_to_idx}")

    _loader = build_dataloader("train", batch_size=8, num_workers=0)
    _x, _y = next(iter(_loader))
    print(f"batch x={tuple(_x.shape)} dtype={_x.dtype}  "
          f"y={tuple(_y.shape)} labels={sorted(set(_y.tolist()))}")
