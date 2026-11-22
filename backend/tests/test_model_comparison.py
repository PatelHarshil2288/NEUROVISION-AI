"""Tests for model comparison service."""

import json
from pathlib import Path
from datetime import datetime, timezone
import math

import pytest

from neurovision_api.services.model_comparison import (
    ModelMetadata,
    ModelComparisonStore,
    ModelComparison,
)
from neurovision_api.services.analytics import EvaluationReport, Misclassification


def compute_metrics_from_matrix(confusion_matrix):
    """Compute precision, recall, f1 from confusion matrix."""
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(10):
        tp = confusion_matrix[i][i]
        # Predicted as class i
        predicted_i = sum(confusion_matrix[row][i] for row in range(10))
        # True class i
        support_i = sum(confusion_matrix[i])
        
        precision = tp / predicted_i if predicted_i > 0 else 0
        recall = tp / support_i if support_i > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return precision_per_class, recall_per_class, f1_per_class


@pytest.fixture
def sample_evaluation_report_a():
    """Create a sample evaluation report for testing."""
    # Simplified confusion matrix - all correct predictions to avoid misclassification complexity
    confusion_matrix = [
        [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1000, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1000, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1000, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1000, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1000, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000],
    ]
    
    correct = sum(confusion_matrix[i][i] for i in range(10))
    accuracy = correct / 10000
    
    support_per_class = [sum(confusion_matrix[i]) for i in range(10)]
    precision_per_class, recall_per_class, f1_per_class = compute_metrics_from_matrix(confusion_matrix)
    
    return EvaluationReport(
        schema_version=1,
        model_version="v0.1.0",
        checkpoint_sha256="a" * 64,
        dataset="MNIST",
        dataset_split="official-test",
        dataset_images_sha256="b" * 64,
        dataset_labels_sha256="c" * 64,
        generated_at=datetime.now(timezone.utc).isoformat(),
        loss=0.08,
        accuracy=accuracy,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        support_per_class=support_per_class,
        confusion_matrix=confusion_matrix,
        misclassified_examples=[],
        num_samples=10000,
    )


@pytest.fixture
def sample_evaluation_report_b():
    """Create a second sample evaluation report with improved metrics."""
    confusion_matrix = [
        [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1000, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1000, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1000, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1000, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1000, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000],
    ]
    
    correct = sum(confusion_matrix[i][i] for i in range(10))
    accuracy = correct / 10000
    
    support_per_class = [sum(confusion_matrix[i]) for i in range(10)]
    precision_per_class, recall_per_class, f1_per_class = compute_metrics_from_matrix(confusion_matrix)
    
    return EvaluationReport(
        schema_version=1,
        model_version="v0.2.0",
        checkpoint_sha256="d" * 64,
        dataset="MNIST",
        dataset_split="official-test",
        dataset_images_sha256="b" * 64,
        dataset_labels_sha256="c" * 64,
        generated_at=datetime.now(timezone.utc).isoformat(),
        loss=0.065,
        accuracy=accuracy,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        support_per_class=support_per_class,
        confusion_matrix=confusion_matrix,
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
    
    # Check accuracy change (may be equal, decreased, or increased)
    accuracy_metric = next(m for m in comparison.metrics if "Accuracy" in m.metric_name)
    assert accuracy_metric.difference <= 0  # Equal or improved in model B (better loss)
    
    # Check loss decrease
    loss_metric = next(m for m in comparison.metrics if "Loss" in m.metric_name)
    assert loss_metric.difference < 0  # Loss should decrease (improvement)


def test_model_comparison_summary(sample_evaluation_report_a, sample_evaluation_report_b):
    """Test that comparison generates meaningful summary."""
    store = ModelComparisonStore(artifacts_dir=Path.cwd())
    
    comparison = store.compare_reports(
        sample_evaluation_report_a,
        sample_evaluation_report_b,
    )
    
    # Summary should mention the models and their differences
    assert comparison.summary is not None
    assert len(comparison.summary) > 50
    # Both models have perfect accuracy, so check for loss difference
    assert "loss" in comparison.summary.lower()


def test_evaluation_reports_persist_and_reject_conflicting_checkpoint_data(
    tmp_path,
    sample_evaluation_report_a,
):
    """Saved reports survive store recreation and remain immutable per checkpoint."""
    store = ModelComparisonStore(artifacts_dir=tmp_path)
    store.save_report(sample_evaluation_report_a)

    restarted_store = ModelComparisonStore(artifacts_dir=tmp_path)
    reports = restarted_store.list_reports()
    assert len(reports) == 1
    assert reports[0].checkpoint_sha256 == sample_evaluation_report_a.checkpoint_sha256
    assert restarted_store.get_report(sample_evaluation_report_a.checkpoint_sha256) == reports[0]

    conflicting_report = sample_evaluation_report_a.model_copy(update={"loss": 0.5})
    with pytest.raises(ValueError, match="different evaluation report"):
        restarted_store.save_report(conflicting_report)
