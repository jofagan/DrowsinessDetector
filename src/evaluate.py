"""Evaluate a trained EyeStateCNN on a data split (default: test).

Run from the project root::

    python -m src.evaluate                                  # test set, default checkpoint
    python -m src.evaluate --split val
    python -m src.evaluate --ckpt models/eyestatecnn.pt --threshold 0.5

Reports loss, overall accuracy, a confusion matrix, and precision/recall/F1 for the
positive class (closed / sleepy = 1), plus per-class accuracy. Metrics are computed by
hand to avoid extra dependencies. The positive class is "closed" because that is the
event the downstream drowsiness alert cares about detecting.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import CLASS_TO_IDX, build_dataloader
from src.model import EyeStateCNN
from src.train import DEFAULT_CKPT, get_device

# Human-readable names indexed by label (0=open/awake, 1=closed/sleepy).
IDX_TO_NAME = {idx: name for name, idx in CLASS_TO_IDX.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


@torch.no_grad()
def evaluate(ckpt_path=DEFAULT_CKPT, split="test", batch_size=64, num_workers=4,
             threshold=0.5, max_batches=0):
    """Evaluate the checkpoint on ``split``; returns a dict of metrics."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with `python -m src.train`."
        )

    device = get_device()
    print(f"Device: {device}")

    model = EyeStateCNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    loader = build_dataloader(split, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.BCEWithLogitsLoss()

    # Confusion counts for positive class = 1 (closed / sleepy).
    tp = tn = fp = fn = 0
    total_loss, total = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        targets = y.float().unsqueeze(1).to(device)

        logits = model(x)
        total_loss += criterion(logits, targets).item() * x.size(0)

        preds = (torch.sigmoid(logits) > threshold).float()
        pos_pred, pos_true = preds == 1, targets == 1
        tp += (pos_pred & pos_true).sum().item()
        tn += (~pos_pred & ~pos_true).sum().item()
        fp += (pos_pred & ~pos_true).sum().item()
        fn += (~pos_pred & pos_true).sum().item()
        total += x.size(0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)          # sensitivity for "closed"
    specificity = _safe_div(tn, tn + fp)     # recall for "open"
    return {
        "split": split,
        "n": total,
        "loss": _safe_div(total_loss, total),
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def print_report(m: dict) -> None:
    """Pretty-print the metrics dict from ``evaluate``."""
    c = m["confusion"]
    open_name, closed_name = IDX_TO_NAME[0], IDX_TO_NAME[1]

    print(f"\nEvaluated {m['n']} {m['split']} images\n")
    print(f"Loss:      {m['loss']:.4f}")
    print(f"Accuracy:  {m['accuracy'] * 100:.2f}%  "
          f"({c['tp'] + c['tn']}/{m['n']})\n")

    print(f"Positive class = closed ({closed_name}, label 1):")
    print(f"  Precision:   {m['precision']:.4f}")
    print(f"  Recall:      {m['recall']:.4f}")
    print(f"  F1:          {m['f1']:.4f}")
    print(f"  Specificity: {m['specificity']:.4f}\n")

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"                    pred open   pred closed")
    print(f"  actual open      {c['tn']:>10}  {c['fp']:>12}")
    print(f"  actual closed    {c['fn']:>10}  {c['tp']:>12}\n")

    open_acc = _safe_div(c["tn"], c["tn"] + c["fp"])
    closed_acc = _safe_div(c["tp"], c["tp"] + c["fn"])
    print("Per-class accuracy:")
    print(f"  open   ({open_name}):  {open_acc * 100:.2f}%")
    print(f"  closed ({closed_name}): {closed_acc * 100:.2f}%")


def main():
    p = argparse.ArgumentParser(description="Evaluate EyeStateCNN on a data split.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability above which an eye is classified closed.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Cap batches for a quick smoke test (0 = full split).")
    args = p.parse_args()

    metrics = evaluate(ckpt_path=args.ckpt, split=args.split, batch_size=args.batch_size,
                       num_workers=args.num_workers, threshold=args.threshold,
                       max_batches=args.max_batches)
    print_report(metrics)


if __name__ == "__main__":
    main()
