from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurovision.renderer.graph import GraphLayout, default_layout, normalize_activation, topk_edges


@dataclass(frozen=True)
class ModelState:
    layout: GraphLayout

    input_activation: np.ndarray  # (784,)
    h1_activation: np.ndarray  # (128,)
    h2_activation: np.ndarray  # (64,)
    out_activation: np.ndarray  # (10,) probabilities

    edges_in_h1: np.ndarray  # (E,2) src=input idx, dst=h1 idx
    edges_h1_h2: np.ndarray  # (E,2) src=h1 idx, dst=h2 idx
    edges_h2_out: np.ndarray  # (E,2) src=h2 idx, dst=out idx


def build_state(
    *,
    image_flat: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    probs: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    max_edges_per_pair: int,
) -> ModelState:
    layout = default_layout()

    input_activation = normalize_activation(image_flat.reshape(-1))
    h1_activation = normalize_activation(h1.reshape(-1))
    h2_activation = normalize_activation(h2.reshape(-1))
    out_activation = np.asarray(probs, dtype=np.float32).reshape(-1)

    # Edge filtering: top-k per destination neuron.
    edges_in_h1 = topk_edges(w1, k=6, max_edges=max_edges_per_pair)
    edges_h1_h2 = topk_edges(w2, k=8, max_edges=max_edges_per_pair)
    edges_h2_out = topk_edges(w3, k=10, max_edges=max_edges_per_pair)

    return ModelState(
        layout=layout,
        input_activation=input_activation,
        h1_activation=h1_activation,
        h2_activation=h2_activation,
        out_activation=out_activation,
        edges_in_h1=edges_in_h1,
        edges_h1_h2=edges_h1_h2,
        edges_h2_out=edges_h2_out,
    )
