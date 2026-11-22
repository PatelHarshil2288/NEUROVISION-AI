from __future__ import annotations

from dataclasses import asdict
from typing import Any

from torch import nn

from neurovision_api.ml.architectures import MNISTConv, MNISTRnn, MNISTTransformer
from neurovision_api.ml.model import MLP, ModelConfig

ARCHITECTURES = ("mlp", "cnn", "rnn", "transformer")


def create_model(architecture: str, config: dict[str, Any] | None = None) -> nn.Module:
    model_config = dict(config or {})
    if architecture == "mlp":
        return MLP(ModelConfig(**model_config))
    if architecture == "cnn":
        return MNISTConv(**model_config)
    if architecture == "rnn":
        return MNISTRnn(**model_config)
    if architecture == "transformer":
        return MNISTTransformer(**model_config)
    raise ValueError(f"Unsupported model architecture: {architecture}.")


def describe_model(model: nn.Module) -> tuple[str, dict[str, Any]]:
    if isinstance(model, MLP):
        return "mlp", asdict(model.config)
    if isinstance(model, MNISTConv):
        return "cnn", {"dropout_p": model.dropout_p}
    if isinstance(model, MNISTRnn):
        return "rnn", {
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "bidirectional": model.bidirectional,
        }
    if isinstance(model, MNISTTransformer):
        return "transformer", {
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "dim_feedforward": model.dim_feedforward,
            "dropout": model.dropout,
        }
    raise ValueError(f"Unsupported model class: {type(model).__name__}.")
