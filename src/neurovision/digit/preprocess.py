from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def canvas_to_mnist(canvas_ink: np.ndarray) -> np.ndarray:
    """Convert a high-resolution canvas to a centered 28x28 MNIST-style image.

    Input: canvas_ink (H,W) float in [0,1] where 1=ink, 0=background.
    Output: (28,28) float32 in [0,1].

    Notes:
    - Crops the ink, preserves its aspect ratio, and fits it into 20x20.
    - Places the resized ink inside a 28x28 canvas with MNIST-like padding.
    - Keeps the pipeline deterministic and CPU-friendly.
    """
    arr = np.asarray(canvas_ink, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D canvas; got shape={arr.shape}')

    arr = np.clip(arr, 0.0, 1.0)
    ink_pixels = arr > 1e-6
    if not np.any(ink_pixels):
        return np.zeros((28, 28), dtype=np.float32)

    rows, cols = np.nonzero(ink_pixels)
    cropped = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = cropped.shape

    scale = min(20.0 / height, 20.0 / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    t = torch.from_numpy(cropped)[None, None, :, :]
    t = F.interpolate(
        t,
        size=(resized_height, resized_width),
        mode='bilinear',
        align_corners=False,
    )
    resized = t[0, 0].cpu().numpy().astype(np.float32)

    out = np.zeros((28, 28), dtype=np.float32)
    row_offset = (28 - resized_height) // 2
    col_offset = (28 - resized_width) // 2
    out[
        row_offset:row_offset + resized_height,
        col_offset:col_offset + resized_width,
    ] = resized

    mx = float(out.max())
    if mx > 1e-6:
        out = out / mx
    return np.clip(out, 0.0, 1.0)
