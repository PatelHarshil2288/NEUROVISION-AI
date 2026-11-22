import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from neurovision_api.ml.checkpoints import load_checkpoint
from neurovision_api.ml.model_factory import create_model
from neurovision_api.ml.model import MLP
from neurovision_api.ml.training import (
    TrainingConfig,
    evaluate_classifier,
    seed_everything,
    train_model,
)
from neurovision_api.services.inference import InferenceService


def test_training_saves_best_and_latest_with_reproducibility_metadata(
    tmp_path,
) -> None:
    seed_everything(42)
    images = torch.randn(24, 1, 28, 28)
    labels = torch.arange(24) % 10
    training_loader = DataLoader(
        TensorDataset(images[:16], labels[:16]),
        batch_size=8,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(images[16:], labels[16:]),
        batch_size=4,
    )

    history = train_model(
        model=MLP(),
        training_loader=training_loader,
        validation_loader=validation_loader,
        checkpoint_dir=tmp_path,
        config=TrainingConfig(epochs=1, batch_size=8),
    )

    best = load_checkpoint(tmp_path / "best.pt")
    latest = load_checkpoint(tmp_path / "latest.pt")
    assert len(history) == 1
    assert best["epoch"] == latest["epoch"] == 1
    assert best["metadata"]["seed"] == 42
    assert best["metadata"]["normalization"] == {
        "mean": 0.1307,
        "standard_deviation": 0.3081,
    }
    assert "optimizer_state" in best


def test_evaluation_returns_per_class_metrics_and_confusion_matrix() -> None:
    model = MLP()
    loader = DataLoader(
        TensorDataset(torch.zeros(10, 1, 28, 28), torch.arange(10)),
        batch_size=5,
    )

    metrics = evaluate_classifier(model, loader, device=torch.device("cpu"))

    assert metrics["num_samples"] == 10
    assert len(metrics["precision_per_class"]) == 10
    assert len(metrics["recall_per_class"]) == 10
    assert len(metrics["f1_per_class"]) == 10
    assert len(metrics["confusion_matrix"]) == 10
    assert len(metrics["misclassified_examples"]) <= 10


@pytest.mark.parametrize(
    ("architecture", "model_config"),
    [
        ("cnn", {"dropout_p": 0.1}),
        ("rnn", {"hidden_size": 4, "num_layers": 1, "bidirectional": False}),
        ("transformer", {
            "d_model": 16, "nhead": 4, "num_layers": 1,
            "dim_feedforward": 32, "dropout": 0.0,
        }),
    ],
)
def test_alternative_architectures_train_and_reload_for_inference(
    tmp_path,
    architecture,
    model_config,
) -> None:
    model = create_model(architecture, model_config)
    images = torch.rand(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    train_model(
        model=model,
        training_loader=loader,
        validation_loader=loader,
        checkpoint_dir=tmp_path,
        config=TrainingConfig(epochs=1, batch_size=2, model_version=f"{architecture}-test"),
    )

    checkpoint = load_checkpoint(tmp_path / "best.pt")
    assert checkpoint["architecture"] == architecture
    service = InferenceService.from_checkpoint(tmp_path / "best.pt")
    result = service.predict([[0.0] * 28 for _ in range(28)])
    assert result["model_version"] == f"{architecture}-test"
    assert len(result["probabilities"]) == 10
