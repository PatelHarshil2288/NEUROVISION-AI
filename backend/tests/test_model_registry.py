import hashlib

import pytest
from fastapi.testclient import TestClient

from neurovision_api.main import create_app
from neurovision_api.ml.checkpoints import save_checkpoint
from neurovision_api.ml.model_factory import create_model, describe_model
from neurovision_api.services.model_comparison import ModelComparisonStore


def _client(app):
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


def _write_checkpoint(path, version, architecture="mlp"):
    model = create_model(architecture)
    architecture, model_config = describe_model(model)
    save_checkpoint(path, {
        "architecture": architecture,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "metadata": {"model_version": version},
        "epoch": 1,
    })
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_activation_rollback_and_restart_persistence(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    first_hash = _write_checkpoint(artifacts_dir / "first.pt", "first")
    second_hash = _write_checkpoint(artifacts_dir / "second.pt", "second", "cnn")

    client = _client(create_app(comparison_artifacts_dir=artifacts_dir))
    available = client.get("/api/v1/model-registry")
    assert available.status_code == 200
    assert {item["checkpoint_hash"] for item in available.json()["available_checkpoints"]} == {
        first_hash,
        second_hash,
    }

    assert client.post("/api/v1/model-registry/rollback").status_code == 409
    unknown = client.post(
        "/api/v1/model-registry/activate",
        json={"checkpoint_hash": "f" * 64},
    )
    assert unknown.status_code == 404

    first = client.post(
        "/api/v1/model-registry/activate",
        json={"checkpoint_hash": first_hash},
    )
    assert first.status_code == 200
    assert first.json()["active"]["checkpoint_hash"] == first_hash
    assert client.get("/api/v1/models").json()["models"][0]["model_version"] == "first"

    second = client.post(
        "/api/v1/model-registry/activate",
        json={"checkpoint_hash": second_hash},
    )
    assert second.status_code == 200
    assert second.json()["active"]["checkpoint_hash"] == second_hash
    assert second.json()["active"]["architecture"] == "cnn"
    prediction = client.post(
        "/api/v1/predict",
        json={"pixels": [[0.0] * 28 for _ in range(28)]},
    )
    assert prediction.json()["model_id"] == "mnist-cnn-v1"
    assert prediction.json()["model_version"] == "second"

    rollback = client.post("/api/v1/model-registry/rollback")
    assert rollback.status_code == 200
    assert rollback.json()["active"]["checkpoint_hash"] == first_hash

    activated_again = client.post(
        "/api/v1/model-registry/activate",
        json={"checkpoint_hash": second_hash},
    )
    assert activated_again.status_code == 200
    restarted_client = _client(create_app(comparison_artifacts_dir=artifacts_dir))
    registry = restarted_client.get("/api/v1/model-registry")
    assert registry.json()["active"]["checkpoint_hash"] == second_hash
    assert restarted_client.get("/api/v1/models").json()["models"][0]["model_version"] == "second"
    assert len(registry.json()["history"]) == 4


def test_model_registry_rejects_forwarded_mutations(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    checkpoint_hash = _write_checkpoint(artifacts_dir / "model.pt", "model")
    client = _client(create_app(comparison_artifacts_dir=artifacts_dir))

    response = client.post(
        "/api/v1/model-registry/activate",
        json={"checkpoint_hash": checkpoint_hash},
        headers={"x-forwarded-for": "192.0.2.1"},
    )
    assert response.status_code == 403


def test_model_registry_rejects_a_second_live_api_worker(tmp_path, monkeypatch):
    store = ModelComparisonStore(artifacts_dir=tmp_path)
    monkeypatch.setattr("neurovision_api.services.model_comparison.os.getpid", lambda: 101)
    store.claim_model_registry_worker()

    monkeypatch.setattr("neurovision_api.services.model_comparison.os.getpid", lambda: 202)
    monkeypatch.setattr(
        "neurovision_api.services.model_comparison.process_is_running",
        lambda process_id: True,
    )
    with pytest.raises(RuntimeError, match="single backend worker"):
        store.claim_model_registry_worker()
