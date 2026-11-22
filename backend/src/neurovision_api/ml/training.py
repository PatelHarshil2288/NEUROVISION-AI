from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from neurovision_api.ml.checkpoints import save_checkpoint
from neurovision_api.ml.model_factory import describe_model


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 128
    seed: int = 42
    model_version: str = "mlp-mnist-v1"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least one.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not self.model_version.strip():
            raise ValueError("model_version cannot be blank.")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class BatchProgress:
    epoch: int
    batch: int
    batches: int
    train_loss: float
    gradient_norm: float
    sample_gradient: float
    sample_before: float
    sample_after: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_loss_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    try:
        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_function(logits, labels)
                batch_size = int(labels.numel())
                total_loss += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += batch_size
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Evaluation data loader is empty.")
    return total_loss / total, correct / total


def train_model(
    *,
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_dir: str | Path,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
    on_batch: Callable[[BatchProgress], None] | None = None,
    on_epoch: Callable[[EpochMetrics, bool], None] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> list[EpochMetrics]:
    training_config = config or TrainingConfig()
    if len(training_loader) == 0:
        raise ValueError("Training data loader is empty.")
    if len(validation_loader) == 0:
        raise ValueError("Validation data loader is empty.")

    target_device = device or torch.device("cpu")
    model.to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    checkpoint_directory = Path(checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    history: list[EpochMetrics] = []
    best_validation_accuracy = -1.0
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        samples_seen = 0
        for batch_index, (images, labels) in enumerate(training_loader, start=1):
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            if on_batch is not None:
                gradient_norm = sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ) ** 0.5
                sample_parameter = next(model.parameters())
                sample_gradient = (
                    float(sample_parameter.grad.detach().reshape(-1)[0].item())
                    if sample_parameter.grad is not None
                    else 0.0
                )
                sample_before = float(sample_parameter.detach().reshape(-1)[0].item())
            optimizer.step()

            batch_size = int(labels.numel())
            cumulative_loss += float(loss.item()) * batch_size
            samples_seen += batch_size
            if on_batch is not None:
                on_batch(
                    BatchProgress(
                        epoch=epoch,
                        batch=batch_index,
                        batches=len(training_loader),
                        train_loss=cumulative_loss / samples_seen,
                        gradient_norm=gradient_norm,
                        sample_gradient=sample_gradient,
                        sample_before=sample_before,
                        sample_after=float(next(model.parameters()).detach().reshape(-1)[0].item()),
                    )
                )

        validation_loss, validation_accuracy = evaluate_loss_accuracy(
            model,
            validation_loader,
            device=target_device,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=cumulative_loss / samples_seen,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        history.append(metrics)

        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            epoch=epoch,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        if checkpoint_metadata is not None:
            checkpoint_payload["metadata"]["experiment"] = checkpoint_metadata
        save_checkpoint(checkpoint_directory / "latest.pt", checkpoint_payload)
        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(checkpoint_directory / "best.pt", checkpoint_payload)
        if on_epoch is not None:
            on_epoch(metrics, improved)

    return history


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
) -> dict[str, Any]:
    architecture, model_config = describe_model(model)
    return {
        "architecture": architecture,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_config": asdict(training_config),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "metadata": {
            "model_version": training_config.model_version,
            "dataset": "MNIST",
            "dataset_version": "MNIST-60k-train-10k-test",
            "normalization": {
                "mean": 0.1307,
                "standard_deviation": 0.3081,
            },
            "seed": training_config.seed,
        },
    }


def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    total_loss = 0.0
    total = 0
    loss_function = nn.CrossEntropyLoss()
    try:
        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_function(logits, labels)
                probabilities = torch.softmax(logits, dim=1)
                batch_predictions = probabilities.argmax(dim=1)
                batch_size = int(labels.numel())
                total_loss += float(loss.item()) * batch_size
                total += batch_size
                targets.extend(int(value) for value in labels.cpu().tolist())
                predictions.extend(
                    int(value) for value in batch_predictions.cpu().tolist()
                )
                confidences.extend(
                    float(value)
                    for value in probabilities.max(dim=1).values.cpu().tolist()
                )
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Evaluation data loader is empty.")

    labels = list(range(10))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    misclassified = [
        {
            "sample_index": index,
            "true_label": target,
            "predicted_label": prediction,
            "confidence": confidences[index],
        }
        for index, (target, prediction) in enumerate(zip(targets, predictions))
        if target != prediction
    ]
    return {
        "loss": total_loss / total,
        "accuracy": sum(
            target == prediction
            for target, prediction in zip(targets, predictions)
        ) / total,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1_score.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=labels,
        ).tolist(),
        "misclassified_examples": misclassified,
        "num_samples": total,
    }
