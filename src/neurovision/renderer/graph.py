from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphLayout:
    input_pos: np.ndarray  # (784, 3)
    h1_pos: np.ndarray  # (128, 3)
    h2_pos: np.ndarray  # (64, 3)
    out_pos: np.ndarray  # (10, 3)


def _grid_positions(rows: int, cols: int, *, x: float, spacing: float = 0.12) -> np.ndarray:
    ys = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2.0) * spacing
    zs = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2.0) * spacing
    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    xx = np.full_like(yy, x, dtype=np.float32)
    pos = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return pos


def default_layout() -> GraphLayout:
    # Input: 28x28 grid
    input_pos = _grid_positions(28, 28, x=-5.0, spacing=0.15)

    # Hidden1: 16x8 (128)
    h1_pos = _grid_positions(16, 8, x=-1.5, spacing=0.22)

    # Hidden2: 8x8 (64)
    h2_pos = _grid_positions(8, 8, x=1.5, spacing=0.25)

    # Output: 10 in a vertical line
    ys = (np.arange(10, dtype=np.float32) - 4.5) * 0.35
    out_pos = np.stack(
        [
            np.full_like(ys, 5.0, dtype=np.float32),
            ys,
            np.zeros_like(ys, dtype=np.float32),
        ],
        axis=-1,
    )

    return GraphLayout(input_pos=input_pos, h1_pos=h1_pos, h2_pos=h2_pos, out_pos=out_pos)


def normalize_activation(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    if v.size == 0:
        return v

    denom = float(np.max(np.abs(v)))
    if denom <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return np.clip(v / denom, -1.0, 1.0).astype(np.float32)


def topk_edges(weights: np.ndarray, k: int, max_edges: int) -> np.ndarray:
    # Return edges as int32 array of shape (E, 2): (src_idx, dst_idx)
    # Picks top-k absolute weights per destination neuron, then caps globally.
    w = np.asarray(weights)
    if w.ndim != 2:
        raise ValueError(f'Expected a 2D weight matrix; got shape={w.shape}')
    if w.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    # weights shape: (dst, src) for torch Linear (out_features, in_features)
    dst, src = w.shape
    k_eff = max(1, min(k, src))

    candidates: list[tuple[float, int, int]] = []
    for j in range(dst):
        row = w[j]
        idx = np.argpartition(np.abs(row), -k_eff)[-k_eff:]
        for i in idx:
            candidates.append((float(abs(row[int(i)])), int(i), int(j)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max(0, max_edges)]
    if not selected:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray([(src_i, dst_i) for _, src_i, dst_i in selected], dtype=np.int32)
