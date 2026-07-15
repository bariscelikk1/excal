"""Small 1D-CNN over windowed pose-feature sequences."""

import torch
import torch.nn as nn

CLASSES = ["jumping_jack", "pull_up", "push_up", "situp", "squat"]


class ExerciseNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int = len(CLASSES)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_features, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features) -> conv over time
        return self.head(self.body(x.transpose(1, 2)))
