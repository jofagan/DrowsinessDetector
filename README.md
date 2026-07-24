Drowsiness Detection

A real-time, fully local drowsiness detector that uses a laptop webcam to classify eye state (open/closed) and alerts when eyes stay closed too long. All processing — face detection, eye cropping, and classification — runs on-device. No frames or data ever leave the machine.

How it works
MediaPipe Face Mesh locates facial landmarks in each webcam frame and crops the eye regions.
A custom-trained PyTorch CNN classifies each cropped eye as open or closed.
If eyes are classified as closed for a sustained number of consecutive frames (rather than a normal blink), the system triggers a drowsiness alert.

Unlike geometric approaches (e.g. Eye Aspect Ratio), this project uses a trained image classifier, so it also generalizes better to variation like glasses, angle, and partial occlusion — at the cost of needing a labeled dataset and a training step.

Project structure
Drowsiness_Detection/
├── data/
│   ├── raw/                # downloaded dataset (CEW and/or MRL), not sorted
│   ├── train/open|closed/
│   ├── val/open|closed/
│   └── test/open|closed/
├── src/
│   ├── sort_dataset.py     # splits data/raw into train/val/test folders
│   ├── dataset.py          # PyTorch Dataset class
│   ├── model.py            # EyeStateCNN architecture
│   ├── train.py            # training loop, saves best weights to models/
│   ├── evaluate.py         # test-set accuracy/metrics
│   └── live_demo.py        # webcam + MediaPipe + trained model, real-time loop
├── models/                 # saved .pt weights (best checkpoint)
├── requirements.txt
└── README.md
Requirements
Python 3.10 or 3.11
A webcam
macOS with Apple Silicon recommended for MPS GPU acceleration (also runs on CPU or CUDA)

Core dependencies (see requirements.txt):

torch, torchvision
opencv-python
mediapipe
numpy, scikit-learn
matplotlib, pillow
Setup
bash
# clone and enter the project
git clone <your-repo-url>
cd Drowsiness_Detection

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
Dataset

This project trains on eye-state image datasets that are not included in the repo — download one (or both) separately:

CEW (Closed Eyes in the Wild) — ~2,400 cropped eye images, roughly balanced open/closed.
MRL Eye Dataset — a larger dataset (~84,000 images) with more variation in lighting, glasses, and angle.

After downloading, place the raw dataset under data/raw/, then run the sorter to split it into train/val/test folders:

bash
python -m src.sort_dataset

This produces a 70/15/15 split into data/train/, data/val/, and data/test/, each with open/ and closed/ subfolders. Spot-check a handful of images in each folder before training — confirm open/ actually contains open eyes and vice versa, since a mislabeled split will train a confidently wrong model.

Usage
1. Train
bash
python -m src.train

Trains the CNN, printing train/val loss and accuracy per epoch. The best checkpoint (by validation loss) is saved to models/.

2. Evaluate
bash
python -m src.evaluate

Runs the saved model against the held-out test set — the final, untouched accuracy/precision/recall numbers.

3. Run the live demo
bash
python -m src.live_demo

Opens the webcam feed, overlays detection, and sounds/displays an alert if eyes stay closed past the duration threshold. Press q or Esc to quit.

macOS camera permission note: if running from an IDE (e.g. PyCharm), camera access is granted per-application. Check System Settings → Privacy & Security → Camera and make sure the IDE (not just Terminal) is enabled, or the webcam feed will silently fail to grab frames.

Results

Fill in with your final numbers after a full training run and test-set evaluation, e.g.:

Metric	Value
Best validation accuracy	—
Test accuracy	—
Test precision / recall	—
Limitations
Eye-shape and lighting variation can affect accuracy; heavier calibration or additional training data may help for edge cases.
Detection quality depends on MediaPipe successfully locating a face — performance degrades if the face is turned away, occluded, or poorly lit.
This is a personal/educational drowsiness detector, not a certified safety device — don't rely on it for real-world driving or safety-critical use.
Acknowledgments
CEW dataset: Song, F., Tan, X., Liu, X., & Chen, S. (2014).
MRL Eye Dataset: Media Research Lab, VSB-TU Ostrava.
