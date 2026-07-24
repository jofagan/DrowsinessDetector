"""EyeStateCNN — small CNN for binary eye-state classification (open vs closed).

Input:  a batch of grayscale eye crops, shape ``(N, 1, 32, 32)``, float tensors
        (pixel values typically scaled to ``[0, 1]``).
Output: raw logits of shape ``(N, 1)`` — no sigmoid is applied, so pair this model
        with ``nn.BCEWithLogitsLoss``. Apply ``torch.sigmoid`` at inference time to
        get probabilities.

Label convention (kept consistent across the pipeline): ``0 = open``, ``1 = closed``.
"""

import torch
import torch.nn as nn


class EyeStateCNN(nn.Module):
    """Two conv blocks (Conv -> ReLU -> MaxPool) followed by a small MLP head.

    Channels double per block starting at 16, so a 32x32 input flows as::

        (N, 1, 32, 32) -> (N, 16, 16, 16) -> (N, 32, 8, 8) -> logits (N, 1)
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels, 32x32 -> 16x16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16 -> 32 channels, 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 1, 32, 32)`` inputs to ``(N, 1)`` raw logits."""
        x = self.features(x)
        return self.classifier(x)
