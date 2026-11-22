from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from neurovision_api.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from neurovision_api.ml.checkpoints import load_checkpoint
from neurovision_api.ml.data import make_mnist_loaders
from neurovision_api.ml.model_factory import ARCHITECTURES, create_model
from neurovision_api.ml.training import (
    TrainingConfig,
    evaluate_classifier,
    seed_everything,
    train_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NEUROVISION MNIST MLP.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="mlp")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        model_version=f"{arguments.architecture}-mnist-v1",
    )
    seed_everything(config.seed)
    loaders = make_mnist_loaders(
        data_dir=arguments.data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    model = create_model(arguments.architecture)
    history = train_model(
        model=model,
        training_loader=loaders.train,
        validation_loader=loaders.validation,
        checkpoint_dir=arguments.checkpoint_dir,
        config=config,
        device=torch.device("cpu"),
    )

    best_checkpoint = load_checkpoint(arguments.checkpoint_dir / "best.pt")
    best_model = create_model(
        best_checkpoint.get("architecture", "mlp"),
        best_checkpoint["model_config"],
    )
    best_model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate_classifier(
        best_model,
        loaders.test,
        device=torch.device("cpu"),
    )
    evaluation_path = arguments.checkpoint_dir / "test-evaluation.json"
    evaluation_path.write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    test_summary = {
        key: test_metrics[key]
        for key in ("loss", "accuracy", "num_samples")
    }
    test_summary["misclassified_count"] = len(
        test_metrics["misclassified_examples"]
    )
    test_summary["misclassified_examples_preview"] = (
        test_metrics["misclassified_examples"][:10]
    )
    print(
        json.dumps(
            {
                "model_version": config.model_version,
                "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
                "selected_epoch": best_checkpoint["epoch"],
                "best_validation_accuracy": best_checkpoint[
                    "validation_accuracy"
                ],
                "history": [asdict(metrics) for metrics in history],
                "test_metrics": test_summary,
                "evaluation_report": str(evaluation_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
