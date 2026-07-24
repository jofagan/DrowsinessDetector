"""Real-time drowsiness demo: webcam -> MediaPipe FaceLandmarker -> EyeStateCNN.

Run from the project root (needs a trained checkpoint and the face-landmark model)::

    python -m src.train                 # produces models/eyestatecnn.pt first
    python -m src.live_demo             # opens the webcam, press 'q' or Esc to quit
    python -m src.live_demo --threshold 0.6 --closed-seconds 1.2 --camera 0

Each frame: FaceLandmarker locates the face, both eye regions are cropped, preprocessed
with the *same* eval transform used in training (grayscale, 32x32, normalized), and scored
by EyeStateCNN. The mean closed-probability decides open vs closed; if the eyes stay closed
continuously for ``--closed-seconds`` a drowsiness alert is shown (and the terminal bell
rings once on onset). Everything runs locally.

MediaPipe here exposes only the Tasks API, so this uses ``FaceLandmarker`` with a local
``face_landmarker.task`` asset (download once; see ``--face-model`` help).
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image

from src.dataset import build_transform
from src.model import EyeStateCNN
from src.train import DEFAULT_CKPT, MODELS_ROOT, get_device

DEFAULT_FACE_MODEL = MODELS_ROOT / "face_landmarker.task"

# MediaPipe Face Mesh eye-contour landmark indices (shared by the Tasks FaceLandmarker).
EYE_LANDMARKS = {
    "right": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}

# The deterministic (no-augmentation) transform, matched to training. Built once.
_EVAL_TRANSFORM = build_transform(train=False)


def eye_bbox(landmarks, indices, width, height, pad_frac=0.4):
    """Pixel bounding box ``(x1, y1, x2, y2)`` around an eye, padded and clamped."""
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return (
        int(max(0, x1 - mx)), int(max(0, y1 - my)),
        int(min(width, x2 + mx)), int(min(height, y2 + my)),
    )


def preprocess_eye(crop_bgr):
    """BGR eye crop -> ``(1, 32, 32)`` tensor via the training-time eval transform."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)  # MRL data was grayscale
    return _EVAL_TRANSFORM(Image.fromarray(gray))      # Grayscale(1) is then a no-op


def classify_eyes(frame_bgr, landmarks, model, device, threshold):
    """Score both eyes; return ``(closed_bool, mean_prob, [(bbox, prob), ...])``.

    Returns ``(None, None, [])`` when no eye crop could be formed.
    """
    height, width = frame_bgr.shape[:2]
    crops, boxes = [], []
    for indices in EYE_LANDMARKS.values():
        x1, y1, x2, y2 = eye_bbox(landmarks, indices, width, height)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size:
            crops.append(preprocess_eye(crop))
            boxes.append((x1, y1, x2, y2))

    if not crops:
        return None, None, []

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()  # closed-prob per eye
    mean_prob = float(probs.mean())
    return mean_prob > threshold, mean_prob, list(zip(boxes, probs))


def load_model(ckpt_path, device):
    """Load EyeStateCNN weights from ``ckpt_path`` onto ``device`` in eval mode."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with `python -m src.train`."
        )
    model = EyeStateCNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def create_landmarker(face_model_path):
    """Create a video-mode FaceLandmarker from a local ``.task`` asset."""
    face_model_path = Path(face_model_path)
    if not face_model_path.exists():
        raise FileNotFoundError(
            f"Face-landmark model not found: {face_model_path}. Download it once with:\n"
            "  curl -fsSL -o models/face_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _draw_overlay(frame, state_text, closed_dur, alert, fps):
    """Draw eye-state text, closed-duration, FPS, and the alert banner in place."""
    color = (0, 0, 255) if (alert or state_text.startswith("CLOSED")) else (0, 255, 0)
    cv2.putText(frame, state_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"closed: {closed_dur:.1f}s   fps: {fps:.0f}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if alert:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 255), -1)
        cv2.putText(frame, "DROWSINESS ALERT", (w // 2 - 150, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


def run(ckpt_path=DEFAULT_CKPT, camera=0, threshold=0.5, closed_seconds=1.5,
        face_model=DEFAULT_FACE_MODEL):
    """Open the webcam and run the real-time detection/alert loop until quit."""
    device = get_device()
    print(f"Device: {device}")
    model = load_model(ckpt_path, device)
    landmarker = create_landmarker(face_model)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera}. Check it is connected and that the "
            "terminal app has camera permission (macOS: System Settings > Privacy > Camera)."
        )

    closed_since = None      # monotonic time eyes first went closed, else None
    alerting = False         # whether we are currently in an alert (for one-shot bell)
    last_ts = -1             # strictly-increasing timestamp for detect_for_video
    prev_t = time.monotonic()
    fps = 0.0
    print("Running. Press 'q' or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break
            frame = cv2.flip(frame, 1)  # mirror for a natural, selfie-view experience

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            closed, state_text = False, "NO FACE"
            if result.face_landmarks:
                closed, mean_prob, per_eye = classify_eyes(
                    frame, result.face_landmarks[0], model, device, threshold)
                if per_eye:
                    for (x1, y1, x2, y2), prob in per_eye:
                        box_color = (0, 0, 255) if prob > threshold else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    state_text = f"{'CLOSED' if closed else 'OPEN'}  p={mean_prob:.2f}"

            now = time.monotonic()
            if closed:
                closed_since = closed_since or now
            else:
                closed_since = None
            closed_dur = (now - closed_since) if closed_since is not None else 0.0
            alert = closed_dur >= closed_seconds

            if alert and not alerting:
                print("\a", end="", flush=True)  # terminal bell on alert onset
            alerting = alert

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            _draw_overlay(frame, state_text, closed_dur, alert, fps)

            cv2.imshow("EyeStateDetector - live", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # 'q' or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


def main():
    p = argparse.ArgumentParser(description="Real-time drowsiness detection demo.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Mean closed-probability above which the eyes count as closed.")
    p.add_argument("--closed-seconds", type=float, default=1.5,
                   help="Continuous closed duration that triggers the alert.")
    p.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL,
                   help="Path to the MediaPipe face_landmarker.task asset.")
    args = p.parse_args()

    run(ckpt_path=args.ckpt, camera=args.camera, threshold=args.threshold,
        closed_seconds=args.closed_seconds, face_model=args.face_model)


if __name__ == "__main__":
    main()
