from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraState:
    yaw_deg: float = 35.0
    pitch_deg: float = -20.0
    distance: float = 14.0


@dataclass(frozen=True)
class RenderConfig:
    # Caps the number of rendered connections per layer-pair.
    max_edges_per_pair: int = 5000
    # Neuron point sizes.
    point_size_input: float = 4.0
    point_size_hidden: float = 6.0
    point_size_output: float = 8.0
