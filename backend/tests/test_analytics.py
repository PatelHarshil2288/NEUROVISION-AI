import json
import sqlite3
from dataclasses import asdict

import pytest
import torch
from fastapi.testclient import TestClient

from neurovision_api.main import create_app
from neurovision_api.ml.checkpoints import save_checkpoint
from neurovision_api.ml.model import MLP
from neurovision_api.services import analytics
from neurovision_api.services.analytics import EvaluationReport
from neurovision_api.services.inference import InferenceService


@pytest.fixture
def setup_analytics(tmp_path):
    model = MLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc1.weight[0, 0] = 2
        model.fc1.bias[0] = 1
        model.fc2.weight[0, 0] = 3
        model.fc2.bias[0] = 1
        model.fc3.weight[4, 0] = 5
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    checkpoint = artifacts_dir / "best.pt"
    save_checkpoint(checkpoint, {
        "model_config": asdict(model.config), "model_state": model.state_dict(),
        "epoch": 1, "metadata": {"model_version": "test-version"},
    })
    service = InferenceService.from_checkpoint(checkpoint)
    matrix = [[0] * 10 for _ in range(10)]
    matrix[0][0] = 1
    matrix[1][2] = 1
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "MNIST" / "raw"
    raw_dir.mkdir(parents=True)
    images_path = raw_dir / "t10k-images-idx3-ubyte"
    labels_path = raw_dir / "t10k-labels-idx1-ubyte"
    images_path.write_bytes(b"test-images")
    labels_path.write_bytes(b"test-labels")
    report = {
        "schema_version": 1, "model_version": service.model_version,
        "checkpoint_sha256": service.checkpoint_sha256,
        "dataset": "MNIST", "dataset_split": "official-test",
        "dataset_images_sha256": analytics.file_sha256(images_path),
        "dataset_labels_sha256": analytics.file_sha256(labels_path),
        "generated_at": "2026-01-01T00:00:00Z", "loss": 1.0, "accuracy": 0.5,
        "precision_per_class": [1.0] + [0.0] * 9,
        "recall_per_class": [1.0] + [0.0] * 9,
        "f1_per_class": [1.0] + [0.0] * 9,
        "support_per_class": [1, 1] + [0] * 8,
        "confusion_matrix": matrix, "num_samples": 2,
        "misclassified_examples": [
            {"sample_index": 1, "true_label": 1, "predicted_label": 2, "confidence": 0.8}
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    app = create_app(
        service=service,
        analytics_report_path=report_path,
        analytics_data_dir=data_dir,
        comparison_artifacts_dir=artifacts_dir,
    )
    app.state.model_comparison_store.save_report(EvaluationReport.model_validate(report))
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
    return client, service, report_path, report


def test_saliency_is_raw_pixel_logit_gradient_and_does_not_mutate_model(setup_analytics):
    client, service, _, _ = setup_analytics
    pixels = [[0.5] * 28 for _ in range(28)]
    result = client.post("/api/v1/saliency", json={"pixels": pixels}).json()
    assert result["predicted_class"] == result["target_class"] == 4
    assert result["gradient"][0][0] == pytest.approx(30 / 0.3081, rel=1e-5)
    assert result["gradient"][0][1] == 0
    assert all(parameter.grad is None for parameter in service.model.parameters())
    other = client.post("/api/v1/saliency", json={"pixels": pixels, "target_class": 0}).json()
    assert other["target_class"] == 0
    assert other["gradient"][0][0] == 0
    for bad in (-1, 10, True):
        assert client.post("/api/v1/saliency", json={"pixels": pixels, "target_class": bad}).status_code == 422


def test_evaluation_uses_verified_database_report_not_editable_json(setup_analytics):
    client, _, path, report = setup_analytics
    assert client.get("/api/v1/analytics/evaluation").json()["accuracy"] == 0.5
    report["checkpoint_sha256"] = "d" * 64
    report["accuracy"] = 0.9
    path.write_text(json.dumps(report), encoding="utf-8")
    response = client.get("/api/v1/analytics/evaluation")
    assert response.status_code == 200
    assert response.json()["checkpoint_sha256"] == setup_analytics[1].checkpoint_sha256
    assert response.json()["accuracy"] == 0.5


def test_missing_active_report_is_evaluated_and_persisted(setup_analytics, monkeypatch):
    client, service, _, report = setup_analytics
    store = client.app.state.model_comparison_store
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DELETE FROM evaluation_reports WHERE checkpoint_sha256 = ?", (service.checkpoint_sha256,))
    monkeypatch.setattr(
        "neurovision_api.api.v1.analytics.test_dataset_hashes",
        lambda root: (report["dataset_images_sha256"], report["dataset_labels_sha256"]),
    )
    evaluated = []

    def fake_evaluation(checkpoint_path, data_dir):
        evaluated.append(checkpoint_path)
        return EvaluationReport.model_validate(report)

    monkeypatch.setattr(
        "neurovision_api.api.v1.analytics.evaluate_mnist_checkpoint",
        fake_evaluation,
    )
    response = client.get("/api/v1/analytics/evaluation")
    assert response.status_code == 200
    assert response.json()["checkpoint_sha256"] == service.checkpoint_sha256
    assert len(evaluated) == 1
    assert store.get_report(service.checkpoint_sha256).accuracy == 0.5
    assert client.get("/api/v1/analytics/evaluation").status_code == 200
    assert len(evaluated) == 1


def test_error_image_requires_reported_sample_and_matching_test_data(setup_analytics, monkeypatch):
    client, _, _, _ = setup_analytics

    class FakeMNIST:
        targets = torch.tensor([0, 1])
        data = torch.zeros((2, 28, 28), dtype=torch.uint8)
        def __len__(self):
            return 2

    monkeypatch.setattr(analytics.datasets, "MNIST", lambda **kwargs: FakeMNIST())
    assert client.get("/api/v1/analytics/misclassifications/0").status_code == 404
    response = client.get("/api/v1/analytics/misclassifications/1")
    assert response.status_code == 200
    assert len(response.json()["pixels"]) == 28


def test_error_image_rejects_different_test_data(setup_analytics, monkeypatch):
    client, _, _, _ = setup_analytics
    monkeypatch.setattr(
        "neurovision_api.api.v1.analytics.test_dataset_hashes",
        lambda root: ("f" * 64, "c" * 64),
    )
    monkeypatch.setattr(analytics, "test_dataset_hashes", lambda root: ("f" * 64, "c" * 64))
    response = client.get("/api/v1/analytics/misclassifications/1")
    assert response.status_code == 409
    assert "different MNIST test data" in response.json()["detail"]
