import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient
from torch.utils.data import Dataset

from neurovision_api.main import create_app
from neurovision_api.ml.checkpoints import load_checkpoint
from neurovision_api.services import training_lab
from neurovision_api.services.training_lab import TrainingLab


class SmallMNIST(Dataset):
    def __len__(self):
        return 6000

    def __getitem__(self, index):
        return torch.full((1, 28, 28), (index % 10) / 10), index % 10


def wait_for(lab: TrainingLab, statuses: tuple[str, ...]) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = lab.snapshot()["job"]
        if job and job["status"] in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError("Training job did not reach the expected state.")


def test_real_lab_training_emits_gradients_metrics_and_isolated_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(training_lab, "TRAIN_SAMPLES", 16)
    monkeypatch.setattr(training_lab, "VALIDATION_SAMPLES", 8)
    monkeypatch.setattr(training_lab, "BATCH_SIZE", 8)
    monkeypatch.setattr(training_lab.datasets, "MNIST", lambda **kwargs: SmallMNIST())
    lab = TrainingLab(enabled=True, data_dir=tmp_path, checkpoint_dir=tmp_path / "lab")

    job = lab.start(1)
    finished = wait_for(lab, ("completed", "failed"))

    assert finished["status"] == "completed", finished["error"]
    assert finished["id"] == job["id"]
    assert finished["epoch"] == 1 and finished["batch"] == 2
    assert finished["gradient_norm"] > 0
    assert finished["sample_gradient"] is not None
    assert finished["sample_after"] is not None
    assert len(finished["history"]) == 1
    assert 0 <= finished["validation_accuracy"] <= 1
    checkpoint = load_checkpoint(tmp_path / "lab" / "best.pt")
    assert checkpoint["metadata"]["model_version"] == f"training-lab-{job['id']}"
    assert checkpoint["metadata"]["experiment"]["train_samples"] == 16
    assert checkpoint["metadata"]["experiment"]["test_set_used"] is False
    assert finished["checkpoint"]["best_epoch"] == 1


def test_lab_enforces_concurrency_quota_and_valid_epochs(tmp_path, monkeypatch):
    lab = TrainingLab(enabled=True, data_dir=tmp_path, checkpoint_dir=tmp_path)
    release = threading.Event()

    def hold(job_id, epochs):
        release.wait(10)
        lab._update(job_id, status="completed")

    monkeypatch.setattr(lab, "_run", hold)
    for bad in (0, 4, True):
        with pytest.raises(ValueError):
            lab.start(bad)
    for _ in range(3):
        release.clear()
        lab.start(1)
        with pytest.raises(RuntimeError, match="already running"):
            lab.start(1)
        release.set()
        wait_for(lab, ("completed",))
    with pytest.raises(RuntimeError, match="quota"):
        lab.start(1)
    assert lab.snapshot()["remaining_jobs"] == 0


def test_training_lab_rejects_a_second_backend_worker(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "shared"
    first_worker = TrainingLab(enabled=True, data_dir=tmp_path, checkpoint_dir=checkpoint_dir)
    second_worker = TrainingLab(enabled=True, data_dir=tmp_path, checkpoint_dir=checkpoint_dir)
    monkeypatch.setattr(first_worker, "_run", lambda job_id, epochs: None)
    monkeypatch.setattr(second_worker, "_run", lambda job_id, epochs: None)
    monkeypatch.setattr(training_lab.os, "getpid", lambda: 101)
    first_worker.start(1)

    monkeypatch.setattr(training_lab.os, "getpid", lambda: 202)
    monkeypatch.setattr(
        "neurovision_api.services.training_lab.process_is_running",
        lambda process_id: True,
    )
    with pytest.raises(RuntimeError, match="one backend worker"):
        second_worker.start(1)


def test_local_api_is_opt_in_and_rejects_remote_or_forwarded_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROVISION_TRAINING_ENABLED", raising=False)
    app = create_app(checkpoint_path=tmp_path / "missing.pt")
    local = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1234))
    remote = TestClient(app, base_url="http://127.0.0.1:8000", client=("192.0.2.1", 1234))

    assert local.get("/api/v1/training").json()["enabled"] is False
    assert local.post("/api/v1/training", json={"epochs": 1}).status_code == 403
    assert remote.get("/api/v1/training").status_code == 403
    assert local.get("/api/v1/training", headers={"x-forwarded-for": "192.0.2.1"}).status_code == 403
    assert local.get("/api/v1/training", headers={"host": "evil.example"}).status_code == 403
    assert local.post("/api/v1/training", json={"epochs": 4}).status_code == 422
    assert local.post("/api/v1/training", json={"epochs": True}).status_code == 422
    assert local.post("/api/v1/training", json={"epochs": 1, "extra": 1}).status_code == 422
