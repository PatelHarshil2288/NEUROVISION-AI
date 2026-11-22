from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = BACKEND_ROOT / "data" / "MNIST"
DEFAULT_ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
DEFAULT_CHECKPOINT_PATH = DEFAULT_ARTIFACTS_DIR / "best.pt"


def get_checkpoint_path() -> Path:
    configured_path = os.environ.get("NEUROVISION_MNIST_CHECKPOINT")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_CHECKPOINT_PATH


def get_allowed_origins() -> list[str]:
    configured_origins = os.environ.get(
        "NEUROVISION_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
