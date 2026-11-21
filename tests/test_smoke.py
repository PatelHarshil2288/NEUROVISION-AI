import numpy as np
import torch
import json
import sys
from torch.utils.data import DataLoader, TensorDataset

from neurovision.digit.preprocess import canvas_to_mnist
from neurovision.ml.checkpoints import load_checkpoint, save_checkpoint
from neurovision.ml.inference import forward_with_telemetry_model
from neurovision.ml.model import MLP, MLPConfig
from neurovision.ml.train import evaluate_accuracy, train_epochs
from neurovision.renderer.graph import topk_edges


def test_canvas_to_mnist_shape_and_range() -> None:
    canvas = np.zeros((280, 280), dtype=np.float32)
    canvas[100:180, 120:160] = 1.0

    mnist = canvas_to_mnist(canvas)
    assert mnist.shape == (28, 28)
    assert float(mnist.min()) >= 0.0
    assert float(mnist.max()) <= 1.0


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = MLPConfig()
    torch.manual_seed(0)
    model = MLP(cfg)

    path = tmp_path / 'ckpt.pt'
    save_checkpoint(path, model=model, config=cfg)

    model2, cfg2 = load_checkpoint(path)
    assert cfg2 == cfg

    x = np.zeros((28, 28), dtype=np.float32)
    r1 = forward_with_telemetry_model(model, x)
    r2 = forward_with_telemetry_model(model2, x)
    assert np.allclose(r1.probabilities, r2.probabilities)


def test_train_loop_smoke() -> None:
    # Tiny synthetic dataset to ensure train loop runs fast.
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    train_dl = DataLoader(ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(ds, batch_size=32, shuffle=False)

    cfg = MLPConfig()
    model = MLP(cfg)

    train_epochs(
        model=model,
        train_dl=train_dl,
        test_dl=test_dl,
        device=torch.device('cpu'),
        epochs=1,
        lr=1e-3,
    )

    acc = evaluate_accuracy(model, test_dl, torch.device('cpu'))
    assert 0.0 <= acc <= 1.0


def test_topk_edges_are_globally_capped_by_weight_magnitude() -> None:
    weights = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.9, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    edges = topk_edges(weights, k=2, max_edges=2)

    assert edges.shape == (2, 2)
    assert set(map(tuple, edges.tolist())) == {(0, 0), (0, 1)}


def test_forward_result_shapes_and_probabilities() -> None:
    cfg = MLPConfig()
    torch.manual_seed(0)
    model = MLP(cfg)
    image = np.zeros((28, 28), dtype=np.float32)

    result = forward_with_telemetry_model(model, image)

    assert result.logits.shape == (10,)
    assert result.probabilities.shape == (10,)
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert result.image_flat.shape == (784,)
    assert result.h1_activation.shape == (128,)
    assert result.h2_activation.shape == (64,)
    assert len(result.events) == 3


def test_training_improves_accuracy_on_separable_digits() -> None:
    torch.manual_seed(7)
    left = torch.zeros(32, 1, 28, 28)
    right = torch.zeros(32, 1, 28, 28)
    left[:, :, 8:20, 4:12] = 1.0
    right[:, :, 8:20, 16:24] = 1.0

    x = torch.cat([left, right])
    y = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
    data = TensorDataset(x, y)
    loader = DataLoader(data, batch_size=16, shuffle=True)

    model = MLP(MLPConfig(output_dim=2))
    train_epochs(
        model=model,
        train_dl=loader,
        test_dl=loader,
        device=torch.device('cpu'),
        epochs=10,
        lr=1e-2,
    )

    accuracy = evaluate_accuracy(model, loader, torch.device('cpu'))
    assert accuracy >= 0.95


def test_diagnostics_mode_returns_json(monkeypatch, capsys) -> None:
    from neurovision.app.main import main

    monkeypatch.setattr(sys, 'argv', ['neurovision', '--diagnostics'])

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload['python']
    assert payload['torch']
    assert isinstance(payload['checkpoint_exists'], bool)
