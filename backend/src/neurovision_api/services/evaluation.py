from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from neurovision_api.ml.data import MNIST_TRANSFORM
from neurovision_api.ml.training import evaluate_classifier
from neurovision_api.services.analytics import EvaluationReport, file_sha256, test_dataset_hashes
from neurovision_api.services.inference import InferenceService


def evaluate_mnist_checkpoint(checkpoint_path: Path, data_dir: Path) -> EvaluationReport:
    checkpoint_hash_before = file_sha256(checkpoint_path)
    service = InferenceService.from_checkpoint(checkpoint_path)
    if service.checkpoint_sha256 != checkpoint_hash_before:
        raise RuntimeError("The checkpoint changed while it was being loaded.")

    dataset_hashes_before = test_dataset_hashes(data_dir)
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=MNIST_TRANSFORM,
        download=False,
    )
    metrics = evaluate_classifier(
        service.model,
        DataLoader(dataset, batch_size=128, shuffle=False),
        device=service.device,
    )
    dataset_hashes_after = test_dataset_hashes(data_dir)
    checkpoint_hash_after = file_sha256(checkpoint_path)
    if dataset_hashes_before != dataset_hashes_after:
        raise RuntimeError("The MNIST test data changed during evaluation; no report was produced.")
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("The checkpoint changed during evaluation; no report was produced.")

    return EvaluationReport(
        schema_version=1,
        model_version=service.model_version,
        checkpoint_sha256=checkpoint_hash_before,
        dataset="MNIST",
        dataset_split="official-test",
        dataset_images_sha256=dataset_hashes_before[0],
        dataset_labels_sha256=dataset_hashes_before[1],
        generated_at=datetime.now(timezone.utc).isoformat(),
        **metrics,
    )
