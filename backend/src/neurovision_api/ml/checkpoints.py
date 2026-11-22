from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(payload)
    checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path | BytesIO,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported or missing checkpoint format version.")
    required_keys = {
        "model_config",
        "model_state",
        "metadata",
        "epoch",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required fields: "
            + ", ".join(sorted(missing_keys))
        )
    return checkpoint
