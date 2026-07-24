# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# EyeStateDetector

A local, real-time drowsiness detector. A webcam feed is passed through MediaPipe
FaceLandmarker to locate/crop both eye regions, and a custom-trained CNN (`EyeStateCNN`)
classifies each eye as open/closed. Sustained closed-eye duration triggers an alert.
Everything runs locally — no cloud calls for inference.

## Stack / environment
- Python **3.14**, virtualenv at `.venv/` (`source .venv/bin/activate`).
- **PyTorch 2.13** + **torchvision 0.28**, with MPS/Apple-Silicon acceleration.
- **mediapipe 0.10.35** for face landmarks; it also provides `cv2` via
  `opencv-contrib-python`. Do **not** also `pip install opencv-python` — two `cv2` variants
  conflict. Plus numpy and Pillow.
- Deps are pinned in `requirements.txt` (`pip install -r requirements.txt`).

## Commands
Run modules from the **project root** so `src` resolves as a (namespace) package — there is
no `__init__.py`, and `python src/train.py` would break the `from src.…` imports:

```bash
python -m src.train                       # train -> models/eyestatecnn.pt (best val acc)
python -m src.train --epochs 20 --lr 5e-4
python -m src.train --epochs 1 --max-batches 15 --num-workers 0   # quick smoke test
python -m src.evaluate                    # test-set loss/acc + confusion matrix + P/R/F1
python -m src.evaluate --split val --threshold 0.6
python -m src.live_demo                   # webcam loop (needs a trained ckpt + camera perm.)
python -m src.dataset                     # dataset smoke test (split sizes, one batch)
```

No test suite or linter is configured; each `src/*.py` has a `__main__` smoke test instead.

## Architecture / data flow
- `src/model.py` — `EyeStateCNN`: 2 conv blocks (16→32 ch) + MLP head. Input `(N,1,32,32)`,
  output raw `(N,1)` logits (no sigmoid; trained with `BCEWithLogitsLoss`).
- `src/dataset.py` — torchvision `ImageFolder` loaders + `build_transform(train)`.
  Normalization constants `MEAN=0.3454, STD=0.1319` (measured on the training split). The
  eval transform is **reused by `live_demo`** so live preprocessing matches training exactly.
- `src/train.py` — training loop; prints per-epoch train/val loss+accuracy, saves the
  best-by-val-accuracy checkpoint. Owns `get_device()` (MPS→CUDA→CPU), reused elsewhere.
- `src/evaluate.py` — metrics on a split; positive class = closed (sleepy). Confusion
  matrix + P/R/F1 computed by hand (no sklearn).
- `src/live_demo.py` — webcam + FaceLandmarker + model, real-time alert loop.

## Data
- MRL infrared **eye-crop** dataset (grayscale PNGs, variable size), pre-split under
  `data/{train,val,test}/{awake,sleepy}/` (~51k train / 17k val / 17k test).
- Dataset docs/helpers live in `data/` (`readme.md`, `get_info.py`, `split_data.py`,
  `labels.txt`) — not part of the `src` pipeline.

## Conventions / gotchas
- Grayscale, 32×32 input throughout (resizing happens in the transform).
- **Label order (the common bug):** `0 = open, 1 = closed`. Folders are `awake`/`sleepy`;
  `ImageFolder` sorts names alphabetically, so `awake`→0 (open) and `sleepy`→1 (closed),
  which matches. `dataset.build_dataset` **asserts** `class_to_idx == {'awake':0,'sleepy':1}`
  so it can't silently flip. Targets for `BCEWithLogitsLoss` are `y.float().unsqueeze(1)`.
- `src/` is a PyCharm Sources Root; internal imports use `from src.<mod> import …`, run via
  `python -m src.<mod>` from the project root.
- **mediapipe** here exposes only the **Tasks API** (`mediapipe.tasks.python.vision`);
  `mp.solutions.*` does not exist. `live_demo` needs the `models/face_landmarker.task` asset
  (one-time download; `live_demo` prints the exact `curl` command if it's missing).
- On macOS, `num_workers > 0` DataLoaders must be built under an `if __name__ == "__main__"`
  guard (train/evaluate already are).

## Not yet decided
- Final `--closed-seconds` and `--threshold` for the alert — tune on the val split once the
  real model has finished training.
