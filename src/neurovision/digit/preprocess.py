from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def canvas_to_mnist(canvas_ink: np.ndarray) -> np.ndarray:
    """Convert a high-res canvas to a 28x28 MNIST-style float32 image.

    Input: canvas_ink (H,W) float in [0,1] where 1=ink, 0=background.
    Output: (28,28) float32 in [0,1].

    Notes:
    - Uses torch interpolation to avoid extra dependencies.
    - Keeps the pipeline deterministic and CPU-friendly.
    """
    arr = np.asarray(canvas_ink, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D canvas; got shape={arr.shape}')

    arr = np.clip(arr, 0.0, 1.0)

    t = torch.from_numpy(arr)[None, None, :, :]
    t = F.interpolate(t, size=(28, 28), mode='bilinear', align_corners=False)
    out = t[0, 0].cpu().numpy().astype(np.float32)

    mx = float(out.max())
    if mx > 1e-6:
        out = out / mx
    return np.clip(out, 0.0, 1.0)
