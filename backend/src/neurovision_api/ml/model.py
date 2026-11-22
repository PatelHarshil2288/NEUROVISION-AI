from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 784
    hidden1: int = 128
    hidden2: int = 64
    output_dim: int = 10

    def __post_init__(self) -> None:
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("All model dimensions must be positive.")
        if self.input_dim != 784:
            raise ValueError("The MNIST MLP requires exactly 784 input features.")
        if self.output_dim != 10:
            raise ValueError("The MNIST MLP requires exactly 10 output classes.")


class MLP(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden1)
        self.fc2 = nn.Linear(self.config.hidden1, self.config.hidden2)
        self.fc3 = nn.Linear(self.config.hidden2, self.config.output_dim)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _flatten_input(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            if tuple(inputs.shape[1:]) != (1, 28, 28):
                raise ValueError(
                    "Image input must have shape (batch, 1, 28, 28); "
                    f"received {tuple(inputs.shape)}."
                )
            inputs = inputs.flatten(start_dim=1)
        elif inputs.ndim != 2:
            raise ValueError(
                "Input must be a flattened batch or a 28x28 image batch; "
                f"received {tuple(inputs.shape)}."
            )

        if inputs.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} features; "
                f"received {inputs.shape[1]}."
            )
        return inputs

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        flattened = self._flatten_input(inputs)
        z1 = self.fc1(flattened)
        a1 = F.relu(z1)
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        logits = self.fc3(a2)
        return {
            "input": flattened,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "logits": logits,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(inputs)["logits"]
