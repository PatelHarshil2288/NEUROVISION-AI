"""Model comparison service for evaluating multiple checkpoint variants."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from neurovision_api.core.processes import process_is_running
from neurovision_api.ml.checkpoints import load_checkpoint
from neurovision_api.ml.model_factory import ARCHITECTURES
from neurovision_api.services.analytics import EvaluationReport

MAX_REPORT_SIZE = 5_000_000


@dataclass
class ModelMetadata:
    """Metadata for a model checkpoint."""
    name: str
    version: str
    checkpoint_path: Path
    report_path: Optional[Path] = None
    checkpoint_sha256: Optional[str] = None
    is_loaded: bool = False

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of checkpoint file."""
        if self.checkpoint_sha256:
            return self.checkpoint_sha256
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()
        return self.checkpoint_sha256


class ModelComparisonMetric(BaseModel):
    """Comparison metric between two models."""
    model_config = ConfigDict(extra="forbid")
    metric_name: str = Field(description="Name of the metric (e.g., accuracy, f1_score)")
    model_a_value: float = Field(description="Metric value for first model")
    model_b_value: float = Field(description="Metric value for second model")
    difference: float = Field(description="Absolute difference (model_b - model_a)")
    percent_change: float = Field(description="Percentage change relative to model_a")


class ModelComparison(BaseModel):
    """Comparison result between two models."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    model_a_name: str = Field(description="Name of first model")
    model_b_name: str = Field(description="Name of second model")
    model_a_version: str = Field(description="Version of first model")
    model_b_version: str = Field(description="Version of second model")
    compared_at: str = Field(description="ISO timestamp of comparison")
    metrics: list[ModelComparisonMetric] = Field(description="Per-metric comparisons")
    summary: str = Field(description="Human-readable summary of key differences")


class AvailableCheckpoint(BaseModel):
    checkpoint_hash: str
    model_version: str
    architecture: str
    checkpoint_name: str
    is_loaded: bool


class ModelActivation(BaseModel):
    checkpoint_hash: str
    model_version: str
    architecture: str
    checkpoint_name: str
    activated_at: str


class ModelComparisonStore:
    """Persist validated model evaluations and comparisons in SQLite."""

    def __init__(self, *, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.database_path = artifacts_dir / "comparison.sqlite3"
        self.models: dict[str, ModelMetadata] = {}
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_reports (
                    checkpoint_sha256 TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    dataset_images_sha256 TEXT NOT NULL,
                    dataset_labels_sha256 TEXT NOT NULL,
                    payload TEXT NOT NULL
                        CHECK (length(CAST(payload AS BLOB)) <= 5000000)
                );
                CREATE TABLE IF NOT EXISTS comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    compared_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                        CHECK (length(CAST(payload AS BLOB)) <= 5000000)
                );
                CREATE INDEX IF NOT EXISTS comparisons_compared_at
                    ON comparisons(compared_at DESC);
                CREATE TABLE IF NOT EXISTS active_model (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    checkpoint_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    checkpoint_name TEXT NOT NULL,
                    activated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_activation_history (
                    activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    checkpoint_name TEXT NOT NULL,
                    activated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS model_activation_history_time
                    ON model_activation_history(activation_id DESC);
                CREATE TABLE IF NOT EXISTS model_registry_process (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    owner_pid INTEGER NOT NULL
                );
                """
            )

    def claim_model_registry_worker(self) -> None:
        """Fail closed if another live API process shares this local model registry."""
        process_id = os.getpid()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_pid FROM model_registry_process WHERE singleton = 1"
            ).fetchone()
            if row is not None and row["owner_pid"] != process_id and process_is_running(row["owner_pid"]):
                raise RuntimeError(
                    "Model registry requires a single backend worker per artifacts directory."
                )
            connection.execute(
                """
                INSERT INTO model_registry_process (singleton, owner_pid) VALUES (1, ?)
                ON CONFLICT(singleton) DO UPDATE SET owner_pid = excluded.owner_pid
                """,
                (process_id,),
            )

    def list_checkpoints(self, *, loaded_checkpoint_hash: str | None = None) -> list[AvailableCheckpoint]:
        """List checkpoint files from approved artifact locations only."""
        root = self.artifacts_dir.resolve()
        candidates = list(root.glob("*.pt"))
        training_lab_dir = root / "training-lab"
        models_dir = root / "models"
        if training_lab_dir.is_dir():
            candidates.extend(training_lab_dir.glob("*.pt"))
        if models_dir.is_dir():
            candidates.extend(models_dir.rglob("*.pt"))

        checkpoints: list[AvailableCheckpoint] = []
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(root):
                continue
            payload = load_checkpoint(resolved_path, map_location="cpu")
            metadata = payload["metadata"]
            if not isinstance(metadata, dict) or not isinstance(metadata.get("model_version"), str):
                raise ValueError(f"Checkpoint metadata is invalid: {path.name}.")
            architecture = payload.get("architecture", "mlp")
            if architecture not in ARCHITECTURES:
                raise ValueError(f"Checkpoint architecture is unsupported: {path.name}.")
            checkpoint_hash = self._file_hash(resolved_path)
            checkpoints.append(AvailableCheckpoint(
                checkpoint_hash=checkpoint_hash,
                model_version=metadata["model_version"],
                architecture=architecture,
                checkpoint_name=resolved_path.relative_to(root).as_posix(),
                is_loaded=checkpoint_hash == loaded_checkpoint_hash,
            ))
        return sorted(checkpoints, key=lambda checkpoint: checkpoint.checkpoint_name)

    def checkpoint_path(self, checkpoint_hash: str) -> Path:
        """Resolve a checkpoint hash only among approved local artifacts."""
        checkpoint = self.checkpoint(checkpoint_hash)
        return self.artifacts_dir / checkpoint.checkpoint_name

    def checkpoint(self, checkpoint_hash: str) -> AvailableCheckpoint:
        """Find checkpoint metadata by hash among approved local artifacts."""
        for checkpoint in self.list_checkpoints():
            if checkpoint.checkpoint_hash == checkpoint_hash:
                return checkpoint
        raise FileNotFoundError("No approved local checkpoint matches this SHA-256 hash.")

    def active_model(self) -> ModelActivation | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                FROM active_model WHERE singleton = 1
                """
            ).fetchone()
        if row is None:
            return None
        return self._activation_from_row(row)

    def initialize_active_model(self, checkpoint: AvailableCheckpoint) -> ModelActivation:
        """Record the configured checkpoint as the initial active model once."""
        activated_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at FROM active_model WHERE singleton = 1"
            ).fetchone()
            if row is not None:
                return self._activation_from_row(row)
            values = (
                checkpoint.checkpoint_hash,
                checkpoint.model_version,
                checkpoint.architecture,
                checkpoint.checkpoint_name,
            )
            connection.execute(
                """
                INSERT INTO active_model (
                    singleton, checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                (*values, activated_at),
            )
            connection.execute(
                """
                INSERT INTO model_activation_history (
                    checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (*values, activated_at),
            )
        return ModelActivation(
            checkpoint_hash=checkpoint.checkpoint_hash,
            model_version=checkpoint.model_version,
            architecture=checkpoint.architecture,
            checkpoint_name=checkpoint.checkpoint_name,
            activated_at=activated_at,
        )

    def activate_model(self, checkpoint: AvailableCheckpoint) -> ModelActivation:
        """Persist an explicit model activation and append it to the history."""
        activated_at = datetime.now(timezone.utc).isoformat()
        values = (
            checkpoint.checkpoint_hash,
            checkpoint.model_version,
            checkpoint.architecture,
            checkpoint.checkpoint_name,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "                SELECT checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at FROM active_model WHERE singleton = 1"
            ).fetchone()
            if current is not None and current["checkpoint_sha256"] == checkpoint.checkpoint_hash:
                return self._activation_from_row(current)
            connection.execute(
                """
                INSERT INTO active_model (
                    singleton, checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    checkpoint_sha256 = excluded.checkpoint_sha256,
                    model_version = excluded.model_version,
                    architecture = excluded.architecture,
                    checkpoint_name = excluded.checkpoint_name,
                    activated_at = excluded.activated_at
                """,
                (*values, activated_at),
            )
            connection.execute(
                """
                INSERT INTO model_activation_history (
                    checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (*values, activated_at),
            )
        return ModelActivation(
            checkpoint_hash=checkpoint.checkpoint_hash,
            model_version=checkpoint.model_version,
            architecture=checkpoint.architecture,
            checkpoint_name=checkpoint.checkpoint_name,
            activated_at=activated_at,
        )

    def previous_model_activation(self) -> ModelActivation:
        with self._connection() as connection:
            current = connection.execute(
                "SELECT checkpoint_sha256 FROM active_model WHERE singleton = 1"
            ).fetchone()
            if current is None:
                raise FileNotFoundError("No active model is registered.")
            row = connection.execute(
                """
                SELECT checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                FROM model_activation_history
                WHERE checkpoint_sha256 != ?
                ORDER BY activation_id DESC LIMIT 1
                """,
                (current["checkpoint_sha256"],),
            ).fetchone()
        if row is None:
            raise FileNotFoundError("No previous model activation is available to roll back to.")
        return self._activation_from_row(row)

    def list_model_activations(self, *, limit: int = 50) -> list[ModelActivation]:
        if not 1 <= limit <= 100:
            raise ValueError("Activation history limit must be from 1 to 100.")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT checkpoint_sha256, model_version, architecture, checkpoint_name, activated_at
                FROM model_activation_history
                ORDER BY activation_id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._activation_from_row(row) for row in rows]

    @staticmethod
    def _activation_from_row(row: sqlite3.Row) -> ModelActivation:
        return ModelActivation(
            checkpoint_hash=row["checkpoint_sha256"],
            model_version=row["model_version"],
            architecture=row["architecture"],
            checkpoint_name=row["checkpoint_name"],
            activated_at=row["activated_at"],
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_report(self, report: EvaluationReport) -> EvaluationReport:
        """Persist a validated evaluation report keyed by checkpoint hash."""
        payload = report.model_dump_json()
        if len(payload.encode("utf-8")) > MAX_REPORT_SIZE:
            raise ValueError("The evaluation report is too large to store.")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload FROM evaluation_reports WHERE checkpoint_sha256 = ?",
                (report.checkpoint_sha256,),
            ).fetchone()
            if existing is not None:
                try:
                    stored_report = EvaluationReport.model_validate_json(existing["payload"])
                except ValueError as exc:
                    raise ValueError("The stored evaluation report is invalid.") from exc
                existing_data = stored_report.model_dump(exclude={"generated_at"})
                incoming_data = report.model_dump(exclude={"generated_at"})
                if existing_data != incoming_data:
                    raise ValueError("A different evaluation report already exists for this checkpoint.")
                return stored_report
            connection.execute(
                """
                INSERT INTO evaluation_reports (
                    checkpoint_sha256, model_version, generated_at,
                    dataset_images_sha256, dataset_labels_sha256, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.checkpoint_sha256,
                    report.model_version,
                    report.generated_at,
                    report.dataset_images_sha256,
                    report.dataset_labels_sha256,
                    payload,
                ),
            )
        return report

    def get_report(self, checkpoint_hash: str) -> EvaluationReport:
        """Load a persisted report by checkpoint SHA-256."""
        if len(checkpoint_hash) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_hash):
            raise ValueError("Checkpoint hash must be a lowercase SHA-256 digest.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM evaluation_reports WHERE checkpoint_sha256 = ?",
                (checkpoint_hash,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No evaluation report is stored for checkpoint {checkpoint_hash}.")
        try:
            return EvaluationReport.model_validate_json(row["payload"])
        except ValueError as exc:
            raise ValueError("The stored evaluation report is invalid.") from exc

    def list_reports(self) -> list[EvaluationReport]:
        """Load all persisted evaluation reports."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM evaluation_reports ORDER BY generated_at DESC, checkpoint_sha256"
            ).fetchall()
        try:
            return [EvaluationReport.model_validate_json(row["payload"]) for row in rows]
        except ValueError as exc:
            raise ValueError("A stored evaluation report is invalid.") from exc

    def save_comparison(self, comparison: ModelComparison) -> ModelComparison:
        """Persist a comparison result for later retrieval."""
        payload = comparison.model_dump_json()
        if len(payload.encode("utf-8")) > MAX_REPORT_SIZE:
            raise ValueError("The saved comparison is too large.")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO comparisons (comparison_id, compared_at, payload) VALUES (?, ?, ?)",
                (comparison.comparison_id, comparison.compared_at, payload),
            )
        return comparison

    def list_comparisons(self) -> list[ModelComparison]:
        """Load saved comparison results, newest first."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM comparisons ORDER BY compared_at DESC, comparison_id"
            ).fetchall()
        try:
            return [ModelComparison.model_validate_json(row["payload"]) for row in rows]
        except ValueError as exc:
            raise ValueError("A stored comparison is invalid.") from exc

    def register_model(
        self,
        name: str,
        version: str,
        checkpoint_path: Path,
        report_path: Optional[Path] = None,
        is_loaded: bool = False,
    ) -> None:
        """Register a model for comparison."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = ModelMetadata(
            name=name,
            version=version,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            is_loaded=is_loaded,
        )
        self.models[f"{name}_{version}"] = model

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models."""
        return list(self.models.values())

    def compare_reports(
        self,
        report_a: EvaluationReport,
        report_b: EvaluationReport,
        name_a: str = "Model A",
        name_b: str = "Model B",
    ) -> ModelComparison:
        """Compare two evaluation reports."""
        if (
            report_a.dataset != report_b.dataset
            or report_a.dataset_split != report_b.dataset_split
            or report_a.dataset_images_sha256 != report_b.dataset_images_sha256
            or report_a.dataset_labels_sha256 != report_b.dataset_labels_sha256
            or report_a.num_samples != report_b.num_samples
        ):
            raise ValueError("Evaluation reports must use the same dataset and split.")

        metrics = []

        # Overall accuracy
        accuracy_diff = report_b.accuracy - report_a.accuracy
        accuracy_pct = (accuracy_diff / report_a.accuracy * 100) if report_a.accuracy else 0
        metrics.append(ModelComparisonMetric(
            metric_name="Overall Accuracy",
            model_a_value=report_a.accuracy,
            model_b_value=report_b.accuracy,
            difference=accuracy_diff,
            percent_change=accuracy_pct,
        ))

        # Overall loss
        loss_diff = report_b.loss - report_a.loss
        loss_pct = (loss_diff / report_a.loss * 100) if report_a.loss else 0
        metrics.append(ModelComparisonMetric(
            metric_name="Loss",
            model_a_value=report_a.loss,
            model_b_value=report_b.loss,
            difference=loss_diff,
            percent_change=loss_pct,
        ))

        # Macro F1 scores
        macro_f1_a = sum(report_a.f1_per_class) / len(report_a.f1_per_class)
        macro_f1_b = sum(report_b.f1_per_class) / len(report_b.f1_per_class)
        f1_diff = macro_f1_b - macro_f1_a
        f1_pct = (f1_diff / macro_f1_a * 100) if macro_f1_a else 0
        metrics.append(ModelComparisonMetric(
            metric_name="Macro F1-Score",
            model_a_value=macro_f1_a,
            model_b_value=macro_f1_b,
            difference=f1_diff,
            percent_change=f1_pct,
        ))

        # Build summary
        accuracy_improvement = (
            "improved" if accuracy_diff > 0 else "declined" if accuracy_diff < 0 else "unchanged"
        )
        loss_change = "decreased" if loss_diff < 0 else "increased" if loss_diff > 0 else "was unchanged"
        summary = (
            f"{name_b} {accuracy_improvement} accuracy by {abs(accuracy_pct):.2f}% "
            f"({report_b.accuracy:.4f} vs {report_a.accuracy:.4f}). "
            f"Loss {loss_change} by {abs(loss_diff):.4f}."
        )

        return ModelComparison(
            model_a_name=name_a,
            model_b_name=name_b,
            model_a_version=report_a.model_version,
            model_b_version=report_b.model_version,
            compared_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            summary=summary,
        )
