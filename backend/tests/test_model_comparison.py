"""Tests for model comparison service."""

import json
from pathlib import Path
from datetime import datetime

import pytest

from neurovision_api.services.model_comparison import (
    ModelMetadata,
    ModelComparisonStore,
    ModelComparison,
)
from neurovision_api.services.analytics import EvaluationReport, Misclassification


@pytest.fixture
def sample_evaluation_report_a():
    """Create a sample evaluation report for testing."""
    return EvaluationReport(
        schema_version=1,
        model_version="v0.1.0",
        checkpoint_sha256="a" * 64,
        dataset="MNIST",
        dataset_split="official-test",
        dataset_images_sha256="b" * 64,
        dataset_labels_sha256="c" * 64,
        generated_at=datetime.utcnow().isoformat() + "Z",
        loss=0.08,
        accuracy=0.9757,
        precision_per_class=[0.98, 0.99, 0.96, 0.98, 0.98, 0.95, 0.97, 0.96, 0.98, 0.97],
        recall_per_class=[0.98, 0.99, 0.95, 0.98, 0.98, 0.95, 0.97, 0.96, 0.98, 0.98],
        f1_per_class=[0.98, 0.99, 0.95, 0.98, 0.98, 0.95, 0.97, 0.96, 0.98, 0.97],
        support_per_class=[980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009],
        confusion_matrix=[
            [960, 0, 2, 0, 0, 1, 10, 1, 6, 0],
            [0, 1125, 2, 2, 0, 1, 1, 2, 2, 0],
            [3, 1, 976, 10, 2, 0, 1, 20, 19, 0],
            [1, 1, 5, 989, 0, 1, 0, 2, 11, 0],
            [1, 0, 2, 1, 963, 0, 7, 1, 1, 6],
            [4, 0, 0, 10, 2, 847, 10, 0, 19, 0],
            [8, 2, 2, 1, 2, 7, 927, 0, 9, 0],
            [1, 7, 20, 2, 1, 1, 0, 982, 2, 12],
            [3, 2, 4, 5, 2, 7, 5, 2, 936, 8],
            [5, 0, 1, 7, 10, 2, 1, 15, 1, 967],
        ],
        misclassified_examples=[
            Misclassification(
                sample_index=0,
                true_label=0,
                predicted_label=4,
                confidence=0.65,
            )
        ],
        num_samples=10000,
    )


@pytest.fixture
def sample_evaluation_report_b():
    """Create a second sample evaluation report with improved metrics."""
    return EvaluationReport(
        schema_version=1,
        model_version="v0.2.0",
        checkpoint_sha256="d" * 64,
        dataset="MNIST",
        dataset_split="official-test",
        dataset_images_sha256="b" * 64,
        dataset_labels_sha256="c" * 64,
        generated_at=datetime.utcnow().isoformat() + "Z",
        loss=0.065,
        accuracy=0.9823,
        precision_per_class=[0.985, 0.995, 0.973, 0.985, 0.985, 0.962, 0.978, 0.973, 0.985, 0.978],
        recall_per_class=[0.985, 0.995, 0.972, 0.985, 0.985, 0.962, 0.978, 0.973, 0.985, 0.985],
        f1_per_class=[0.985, 0.995, 0.972, 0.985, 0.985, 0.962, 0.978, 0.973, 0.985, 0.981],
        support_per_class=[980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009],
        confusion_matrix=[
            [965, 0, 1, 0, 0, 1, 7, 0, 5, 1],
            [0, 1130, 1, 1, 0, 1, 1, 1, 0, 0],
            [2, 0, 1002, 7, 1, 0, 1, 15, 4, 0],
            [1, 0, 3, 994, 0, 1, 0, 1, 10, 0],
            [0, 0, 1, 0, 965, 0, 6, 0, 0, 10],
            [3, 0, 0, 7, 1, 858, 8, 0, 15, 0],
            [6, 1, 1, 1, 1, 5, 938, 0, 5, 0],
            [1, 5, 15, 2, 1, 1, 0, 1000, 1, 2],
            [2, 1, 3, 4, 1, 5, 4, 1, 950, 3],
            [4, 0, 1, 5, 8, 2, 1, 10, 1, 977],
        ],
        misclassified_examples=[],
        num_samples=10000,
    )


def test_model_metadata_hash_computation(tmp_path):
    """Test that model metadata can compute checkpoint hash."""
    checkpoint_file = tmp_path / "model.pt"
    checkpoint_file.write_text("dummy checkpoint")
    
    model = ModelMetadata(
        name="test",
        version="v1.0",
        checkpoint_path=checkpoint_file,
    )
    
    hash_val = model.compute_hash()
    assert len(hash_val) == 64
    assert hash_val.isalnum()


def test_model_comparison_store_registration(tmp_path):
    """Test model registration in comparison store."""
    checkpoint_file = tmp_path / "model.pt"
    checkpoint_file.write_text("dummy")
    
    store = ModelComparisonStore(artifacts_dir=tmp_path)
    store.register_model("model", "v1", checkpoint_file)
    
    models = store.list_models()
    assert len(models) == 1
    assert models[0].name == "model"
    assert models[0].version == "v1"


def test_model_comparison_reports(sample_evaluation_report_a, sample_evaluation_report_b):
    """Test comparison of two evaluation reports."""
    store = ModelComparisonStore(artifacts_dir=Path.cwd())
    
    comparison = store.compare_reports(
        sample_evaluation_report_a,
        sample_evaluation_report_b,
        name_a="MNIST MLP v1",
        name_b="MNIST MLP v2",
    )
    
    assert isinstance(comparison, ModelComparison)
    assert comparison.model_a_name == "MNIST MLP v1"
    assert comparison.model_b_name == "MNIST MLP v2"
    assert len(comparison.metrics) == 3
    
    # Check accuracy improvement
    accuracy_metric = next(m for m in comparison.metrics if "Accuracy" in m.metric_name)
    assert accuracy_metric.difference > 0
    assert accuracy_metric.percent_change > 0
    
    # Check loss decrease
    loss_metric = next(m for m in comparison.metrics if "Loss" in m.metric_name)
    assert loss_metric.difference < 0


def test_model_comparison_summary(sample_evaluation_report_a, sample_evaluation_report_b):
    """Test that comparison generates meaningful summary."""
    store = ModelComparisonStore(artifacts_dir=Path.cwd())
    
    comparison = store.compare_reports(
        sample_evaluation_report_a,
        sample_evaluation_report_b,
    )
    
    assert "improved" in comparison.summary.lower()
    assert comparison.summary is not None
    assert len(comparison.summary) > 50
