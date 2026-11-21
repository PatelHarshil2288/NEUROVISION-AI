from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 784
    hidden1: int = 128
    hidden2: int = 64
    output_dim: int = 10


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.input_dim, config.hidden1)
        self.fc2 = nn.Linear(config.hidden1, config.hidden2)
        self.fc3 = nn.Linear(config.hidden2, config.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
