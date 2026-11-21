from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerEvent:
    layer_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    pre_activation_stats: dict[str, float]
    activation_stats: dict[str, float]


@dataclass(frozen=True)
class ModelRunResult:
    events: list[LayerEvent]
