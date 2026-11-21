from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from neurovision.ml.model import MLP, MLPConfig


@dataclass(frozen=True)
class Checkpoint:
    config: MLPConfig
    state_dict: dict


def save_checkpoint(path: str | Path, *, model: nn.Module, config: MLPConfig) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'config': asdict(config),
        'state_dict': model.state_dict(),
    }
    torch.save(payload, p)


def load_checkpoint(path: str | Path, *, map_location: str = 'cpu') -> tuple[MLP, MLPConfig]:
    p = Path(path)
    payload = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or 'config' not in payload or 'state_dict' not in payload:
        raise ValueError(f'Invalid NEUROVISION checkpoint: {p}')

    cfg = MLPConfig(**payload['config'])
    model = MLP(cfg)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, cfg
