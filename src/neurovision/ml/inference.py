from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neurovision.ml.model import MLP, MLPConfig
from neurovision.telemetry.events import LayerEvent


@dataclass(frozen=True)
class ForwardResult:
    logits: np.ndarray
    probabilities: np.ndarray
    events: list[LayerEvent]

    image_flat: np.ndarray  # (784,)
    h1_activation: np.ndarray  # (128,)
    h2_activation: np.ndarray  # (64,)

    w1: np.ndarray  # (128,784)
    w2: np.ndarray  # (64,128)
    w3: np.ndarray  # (10,64)


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    t = t.detach().float().cpu()
    return {
        'min': float(t.min().item()),
        'max': float(t.max().item()),
        'mean': float(t.mean().item()),
        'std': float(t.std(unbiased=False).item()),
    }


def forward_with_telemetry_model(model: MLP, image_28x28: np.ndarray) -> ForwardResult:
    if image_28x28.shape != (28, 28):
        raise ValueError(f'Expected image shape (28, 28); got {image_28x28.shape}')

    model.eval()

    x = torch.from_numpy(image_28x28.astype(np.float32)).reshape(1, -1)

    events: list[LayerEvent] = []
    with torch.no_grad():
        z1 = model.fc1(x)
        a1 = model.relu(z1)
        events.append(
            LayerEvent(
                layer_name='fc1',
                input_shape=tuple(x.shape),
                output_shape=tuple(a1.shape),
                pre_activation_stats=_tensor_stats(z1),
                activation_stats=_tensor_stats(a1),
            )
        )

        z2 = model.fc2(a1)
        a2 = model.relu(z2)
        events.append(
            LayerEvent(
                layer_name='fc2',
                input_shape=tuple(a1.shape),
                output_shape=tuple(a2.shape),
                pre_activation_stats=_tensor_stats(z2),
                activation_stats=_tensor_stats(a2),
            )
        )

        logits_t = model.fc3(a2)
        probs_t = torch.softmax(logits_t, dim=-1)
        events.append(
            LayerEvent(
                layer_name='fc3',
                input_shape=tuple(a2.shape),
                output_shape=tuple(logits_t.shape),
                pre_activation_stats=_tensor_stats(logits_t),
                activation_stats=_tensor_stats(probs_t),
            )
        )

    logits = logits_t.cpu().numpy().reshape(-1)
    probabilities = probs_t.cpu().numpy().reshape(-1)

    return ForwardResult(
        logits=logits,
        probabilities=probabilities,
        events=events,
        image_flat=x.cpu().numpy().reshape(-1),
        h1_activation=a1.cpu().numpy().reshape(-1),
        h2_activation=a2.cpu().numpy().reshape(-1),
        w1=model.fc1.weight.detach().cpu().numpy(),
        w2=model.fc2.weight.detach().cpu().numpy(),
        w3=model.fc3.weight.detach().cpu().numpy(),
    )


def forward_with_telemetry(image: np.ndarray, config: MLPConfig) -> ForwardResult:
    # Backwards-compatible helper: create a deterministic model.
    torch.manual_seed(0)
    model = MLP(config)
    return forward_with_telemetry_model(model, image)
