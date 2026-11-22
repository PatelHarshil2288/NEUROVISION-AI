"""Integration tests for model comparison API endpoints."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from neurovision_api.main import create_app
from neurovision_api.ml.checkpoints import save_checkpoint
from neurovision_api.ml.model import MLP
from neurovision_api.services.analytics import EvaluationReport
from neurovision_api.services.comparison_migration import migrate_legacy_comparison_json
from neurovision_api.services.model_comparison import ModelComparisonStore


@pytest.fixture
def test_app(tmp_path):
    """Create test application."""
    return create_app(service=None, comparison_artifacts_dir=tmp_path)


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app, base_url="http://localhost", client=("127.0.0.1", 50000))


def test_comparison_summary_endpoint(client):
    """Test model comparison summary endpoint."""
    response = client.get("/api/v1/comparison/summary")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_models" in data
    assert "models_registered" in data
    assert "feature_status" in data
    assert "message" in data
    
    # Check feature status structure
    features = data["feature_status"]
    assert "multi_model_evaluation" in features
    assert "historical_comparison" in features
    assert "version_tracking" in features


def test_available_models_endpoint(client):
    """Test listing available models endpoint."""
    response = client.get("/api/v1/comparison/available-models")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)
    assert data["count"] >= 0


def test_model_comparison_requires_multiple_models(client):
    """Test that model comparison reports missing checkpoints clearly."""
    response = client.post(
        "/api/v1/comparison/reports",
        json={
            "model_a_checkpoint_hash": "a" * 64,
            "model_b_checkpoint_hash": "b" * 64,
        }
    )
    
    assert response.status_code == 404


def _evaluation_report(checkpoint_hash, version, loss=0.1, dataset_hash="b" * 64):
    matrix = [[1000 if row == column else 0 for column in range(10)] for row in range(10)]
    return {
        "schema_version": 1,
        "model_version": version,
        "checkpoint_sha256": checkpoint_hash,
        "dataset": "MNIST",
        "dataset_split": "official-test",
        "dataset_images_sha256": dataset_hash,
        "dataset_labels_sha256": "c" * 64,
        "generated_at": "2026-01-01T00:00:00Z",
        "loss": loss,
        "accuracy": 1.0,
        "precision_per_class": [1.0] * 10,
        "recall_per_class": [1.0] * 10,
        "f1_per_class": [1.0] * 10,
        "support_per_class": [1000] * 10,
        "confusion_matrix": matrix,
        "misclassified_examples": [],
        "num_samples": 10000,
    }


def test_reports_and_comparisons_persist_and_compare(client, test_app, tmp_path):
    """Persist backend-generated reports, compare them, and read after app restart."""
    report_a = _evaluation_report("a" * 64, "model-v1", loss=0.1)
    report_b = _evaluation_report("d" * 64, "model-v2", loss=0.05)

    for report in (report_a, report_b):
        stored = test_app.state.model_comparison_store.save_report(EvaluationReport.model_validate(report))
        assert stored.checkpoint_sha256 == report["checkpoint_sha256"]

    available = client.get("/api/v1/comparison/available-models").json()
    assert available["count"] == 2
    assert {model["checkpoint_hash"] for model in available["models"]} == {"a" * 64, "d" * 64}

    comparison_response = client.post("/api/v1/comparison/reports", json={
        "model_a_checkpoint_hash": "a" * 64,
        "model_b_checkpoint_hash": "d" * 64,
        "model_a_name": "model-v1",
        "model_b_name": "model-v2",
    })
    assert comparison_response.status_code == 200
    comparison = comparison_response.json()
    assert comparison["metrics"][1]["difference"] == pytest.approx(-0.05)
    assert len(client.get("/api/v1/comparison/reports").json()) == 1

    restarted_client = TestClient(
        create_app(service=None, comparison_artifacts_dir=tmp_path),
        base_url="http://localhost",
        client=("127.0.0.1", 50000),
    )
    assert len(restarted_client.get("/api/v1/comparison/evaluation-reports").json()) == 2
    saved_comparisons = restarted_client.get("/api/v1/comparison/reports").json()
    assert saved_comparisons[0]["comparison_id"] == comparison["comparison_id"]


def test_comparison_rejects_reports_from_different_test_data(client):
    """Do not compare reports measured on different MNIST test data."""
    report_a = _evaluation_report("a" * 64, "model-v1")
    report_b = _evaluation_report("d" * 64, "model-v2", dataset_hash="e" * 64)
    for report in (report_a, report_b):
        client.app.state.model_comparison_store.save_report(EvaluationReport.model_validate(report))

    response = client.post("/api/v1/comparison/reports", json={
        "model_a_checkpoint_hash": "a" * 64,
        "model_b_checkpoint_hash": "d" * 64,
    })
    assert response.status_code == 409
    assert "same dataset" in response.json()["detail"]


def test_comparison_endpoints_dont_break_existing_api(client):
    """Ensure new comparison endpoints don't interfere with existing API."""
    # Test that existing health endpoint still works
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    # Test that existing model endpoint still works
    response = client.get("/api/v1/models")
    assert response.status_code == 200


def test_evaluation_report_endpoint_evaluates_only_approved_checkpoints(
    client,
    test_app,
    tmp_path,
    monkeypatch,
):
    model = MLP()
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, {
        "model_config": vars(model.config),
        "model_state": model.state_dict(),
        "metadata": {"model_version": "approved-v1"},
        "epoch": 1,
    })
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    store = test_app.state.model_comparison_store
    store.artifacts_dir = tmp_path
    test_app.state.analytics_store.data_dir = tmp_path / "data"

    def fake_evaluation(path, data_dir):
        assert path == checkpoint_path
        assert data_dir == tmp_path / "data"
        report = _evaluation_report(checkpoint_hash, "approved-v1")
        return EvaluationReport.model_validate(report)

    monkeypatch.setattr(
        "neurovision_api.api.v1.comparison.evaluate_mnist_checkpoint",
        fake_evaluation,
    )

    checkpoints = client.get("/api/v1/comparison/checkpoints")
    assert checkpoints.status_code == 200
    assert [item["checkpoint_hash"] for item in checkpoints.json()] == [checkpoint_hash]

    response = client.post(
        "/api/v1/comparison/evaluation-reports",
        json={"checkpoint_hash": checkpoint_hash},
    )
    assert response.status_code == 201
    assert response.json()["checkpoint_sha256"] == checkpoint_hash

    rejected = client.post(
        "/api/v1/comparison/evaluation-reports",
        json={**_evaluation_report(checkpoint_hash, "approved-v1")},
    )
    assert rejected.status_code == 422

    forwarded = client.post(
        "/api/v1/comparison/evaluation-reports",
        json={"checkpoint_hash": checkpoint_hash},
        headers={"x-forwarded-for": "192.0.2.1"},
    )
    assert forwarded.status_code == 403


def test_legacy_json_migration_re_evaluates_reports_and_preserves_history(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    lab_dir = artifacts_dir / "training-lab"
    legacy_report_dir = artifacts_dir / "evaluation-reports"
    legacy_comparison_dir = artifacts_dir / "comparisons"
    for directory in (lab_dir, legacy_report_dir, legacy_comparison_dir):
        directory.mkdir(parents=True)

    model = MLP()
    checkpoint_paths = (
        artifacts_dir / "best.pt",
        lab_dir / "best.pt",
    )
    versions = ("model-v1", "model-v2")
    reports = []
    for checkpoint_path, version in zip(checkpoint_paths, versions):
        save_checkpoint(checkpoint_path, {
            "model_config": vars(model.config),
            "model_state": model.state_dict(),
            "metadata": {"model_version": version},
            "epoch": 1,
        })
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        reports.append(EvaluationReport.model_validate(
            _evaluation_report(checkpoint_hash, version)
        ))

    store = ModelComparisonStore(artifacts_dir=artifacts_dir)
    for report in reports:
        (legacy_report_dir / f"{report.checkpoint_sha256}.json").write_text(
            report.model_dump_json(),
            encoding="utf-8",
        )
    legacy_comparison = store.compare_reports(
        reports[0],
        reports[1],
        name_a="Model A",
        name_b="Model B",
    ).model_copy(update={
        "comparison_id": "legacy-comparison",
        "compared_at": "2026-01-02T00:00:00Z",
        "metrics": [
            store.compare_reports(reports[0], reports[1]).metrics[0].model_copy(
                update={"difference": 999.0}
            ),
            *store.compare_reports(reports[0], reports[1]).metrics[1:],
        ],
    })
    (legacy_comparison_dir / "legacy-comparison.json").write_text(
        legacy_comparison.model_dump_json(),
        encoding="utf-8",
    )
    evaluated_paths = []

    def evaluator(checkpoint_path, data_dir):
        assert data_dir == tmp_path / "data"
        evaluated_paths.append(checkpoint_path)
        return reports[checkpoint_paths.index(checkpoint_path)]

    migrated_reports, migrated_comparisons = migrate_legacy_comparison_json(
        store,
        data_dir=tmp_path / "data",
        evaluator=evaluator,
    )

    assert (migrated_reports, migrated_comparisons) == (2, 1)
    assert set(evaluated_paths) == set(checkpoint_paths)
    assert len(store.list_reports()) == 2
    migrated = store.list_comparisons()[0]
    assert migrated.comparison_id == "legacy-comparison"
    assert migrated.compared_at == "2026-01-02T00:00:00Z"
    assert migrated.metrics[0].difference != 999.0
