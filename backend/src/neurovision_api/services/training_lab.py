from __future__ import annotations

import copy
import logging
import math
import os
import sqlite3
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from neurovision_api.core.processes import process_is_running
from neurovision_api.ml.data import MNIST_TRANSFORM, split_training_dataset
from neurovision_api.ml.model import MLP
from neurovision_api.ml.training import BatchProgress, EpochMetrics, TrainingConfig, train_model

logger = logging.getLogger(__name__)
MAX_JOBS = 3
TRAIN_SAMPLES = 2048
VALIDATION_SAMPLES = 512
BATCH_SIZE = 128
SEED = 42


class TrainingLab:
    """A bounded, single-process local training laboratory."""

    def __init__(self, *, enabled: bool, data_dir: Path, checkpoint_dir: Path) -> None:
        self.enabled = enabled
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.state_path = checkpoint_dir / "training-lab.sqlite3"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        with sqlite3.connect(self.state_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    owner_pid INTEGER,
                    jobs_started INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO training_state (singleton, owner_pid, jobs_started) VALUES (1, NULL, 0)"
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            with sqlite3.connect(self.state_path) as connection:
                row = connection.execute(
                    "SELECT jobs_started FROM training_state WHERE singleton = 1"
                ).fetchone()
            if row is None:
                raise RuntimeError("Training laboratory state is unavailable.")
            return {
                "enabled": self.enabled,
                "remaining_jobs": max(0, MAX_JOBS - row[0]),
                "job": copy.deepcopy(self._job),
            }

    def start(self, epochs: int) -> dict[str, Any]:
        if not self.enabled:
            raise PermissionError("The training laboratory is disabled.")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 3:
            raise ValueError("Epochs must be an integer from 1 to 3.")
        with self._lock:
            if self._job and self._job["status"] in ("queued", "running"):
                raise RuntimeError("A training job is already running.")
            self._claim_single_worker_job()
            job_id = uuid.uuid4().hex
            self._job = {
                "id": job_id,
                "status": "queued",
                "epochs": epochs,
                "epoch": 0,
                "batch": 0,
                "batches": TRAIN_SAMPLES // BATCH_SIZE,
                "train_loss": None,
                "validation_loss": None,
                "validation_accuracy": None,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "train_samples": TRAIN_SAMPLES,
                "validation_samples": VALIDATION_SAMPLES,
                "gradient_norm": None,
                "sample_gradient": None,
                "sample_before": None,
                "sample_after": None,
                "history": [],
                "checkpoint": None,
                "error": None,
            }
            thread = threading.Thread(target=self._run, args=(job_id, epochs), daemon=True)
            try:
                thread.start()
            except RuntimeError:
                self._job["status"] = "failed"
                self._job["error"] = "The training worker could not start."
                logger.exception("Could not start training worker")
                raise
            return copy.deepcopy(self._job)

    def _claim_single_worker_job(self) -> None:
        process_id = os.getpid()
        connection = sqlite3.connect(self.state_path, timeout=30)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_pid, jobs_started FROM training_state WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("Training laboratory state is unavailable.")
            owner_pid, jobs_started = row
            if owner_pid is not None and owner_pid != process_id and process_is_running(owner_pid):
                raise RuntimeError("Training is restricted to one backend worker process.")
            if jobs_started >= MAX_JOBS:
                raise RuntimeError("The three-job process quota has been reached.")
            connection.execute(
                "UPDATE training_state SET owner_pid = ?, jobs_started = ? WHERE singleton = 1",
                (process_id, jobs_started + 1),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job and self._job["id"] == job_id:
                self._job.update(values)

    def _run(self, job_id: str, epochs: int) -> None:
        try:
            self._update(job_id, status="running")
            dataset = datasets.MNIST(
                root=str(self.data_dir), train=True, transform=MNIST_TRANSFORM, download=False
            )
            train, validation = split_training_dataset(dataset, seed=SEED)
            train_loader = DataLoader(
                Subset(train, range(TRAIN_SAMPLES)),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(SEED),
                num_workers=0,
            )
            validation_loader = DataLoader(
                Subset(validation, range(VALIDATION_SAMPLES)),
                batch_size=BATCH_SIZE,
                num_workers=0,
            )
            model = MLP()
            generator = torch.Generator().manual_seed(SEED)
            with torch.no_grad():
                for layer in (model.fc1, model.fc2, model.fc3):
                    bound = 1 / math.sqrt(layer.in_features)
                    layer.weight.uniform_(-bound, bound, generator=generator)
                    layer.bias.uniform_(-bound, bound, generator=generator)

            def batch_update(progress: BatchProgress) -> None:
                self._update(job_id, **asdict(progress))

            def epoch_update(metrics: EpochMetrics, improved: bool) -> None:
                with self._lock:
                    if self._job and self._job["id"] == job_id:
                        self._job["history"].append(asdict(metrics))
                        self._job["validation_loss"] = metrics.validation_loss
                        self._job["validation_accuracy"] = metrics.validation_accuracy
                        best_epoch = metrics.epoch if improved else self._job["checkpoint"]["best_epoch"]
                        self._job["checkpoint"] = {
                            "latest_epoch": metrics.epoch,
                            "best_epoch": best_epoch,
                            "directory": "artifacts/training-lab",
                        }

            train_model(
                model=model,
                training_loader=train_loader,
                validation_loader=validation_loader,
                checkpoint_dir=self.checkpoint_dir,
                config=TrainingConfig(
                    epochs=epochs, batch_size=BATCH_SIZE, seed=SEED,
                    model_version=f"training-lab-{job_id}",
                ),
                device=torch.device("cpu"),
                on_batch=batch_update,
                on_epoch=epoch_update,
                checkpoint_metadata={
                    "kind": "training-lab",
                    "train_samples": TRAIN_SAMPLES,
                    "validation_samples": VALIDATION_SAMPLES,
                    "selection": "seeded 55000/5000 split; first subset indices",
                    "test_set_used": False,
                },
            )
            self._update(job_id, status="completed")
        except Exception as exc:
            logger.exception("Training lab job %s failed", job_id)
            self._update(job_id, status="failed", error=str(exc) or type(exc).__name__)
