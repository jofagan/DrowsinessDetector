"""Training loop for EyeStateCNN on the MRL eye-state dataset.

Run from the project root so ``src`` resolves as a package::

    python -m src.train                       # 10 epochs, defaults
    python -m src.train --epochs 20 --lr 5e-4
    python -m src.train --epochs 1 --max-batches 15 --num-workers 0   # quick smoke test

Uses the target convention 0=open (awake) / 1=closed (sleepy), ``BCEWithLogitsLoss``
against the model's raw ``(N, 1)`` logits, and saves the best-by-val-accuracy weights to
``models/eyestatecnn.pt``. Trains on Apple-Silicon MPS when available, else CPU.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_dataloader
from src.model import EyeStateCNN

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CKPT = MODELS_ROOT / "eyestatecnn.pt"


def get_device() -> torch.device:
    """Prefer Apple-Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    """Run one pass over ``loader``; returns ``(avg_loss, accuracy)``.

    Passing an ``optimizer`` puts the model in train mode and backprops; otherwise
    the pass is evaluation-only under ``no_grad``.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        # BCEWithLogitsLoss wants float targets shaped like the (N, 1) logits.
        targets = y.float().unsqueeze(1).to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch = x.size(0)
        total_loss += loss.item() * batch
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += batch

    return total_loss / total, correct / total


def train(epochs=10, batch_size=64, lr=1e-3, num_workers=4, seed=42,
          ckpt_path=DEFAULT_CKPT, max_batches=0):
    """Train EyeStateCNN, printing loss/accuracy per epoch and saving the best model."""
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader = build_dataloader("train", batch_size=batch_size, num_workers=num_workers)
    val_loader = build_dataloader("val", batch_size=batch_size, num_workers=num_workers)

    model = EyeStateCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, max_batches)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device,
                                      max_batches=max_batches)
        dt = time.time() - t0

        line = (f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc * 100:5.2f}% | "
                f"val loss {val_loss:.4f} acc {val_acc * 100:5.2f}% | {dt:5.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            line += "  <- best, saved"
        print(line)

    print(f"Best val acc: {best_val_acc * 100:.2f}%  ->  {ckpt_path}")
    return best_val_acc


def main():
    p = argparse.ArgumentParser(description="Train EyeStateCNN on the MRL eye dataset.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--max-batches", type=int, default=0,
                   help="Cap batches per epoch for quick smoke tests (0 = full).")
    args = p.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          num_workers=args.num_workers, seed=args.seed, ckpt_path=args.ckpt,
          max_batches=args.max_batches)


if __name__ == "__main__":
    main()
