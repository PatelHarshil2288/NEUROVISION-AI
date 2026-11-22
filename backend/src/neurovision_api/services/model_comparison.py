"""Model comparison service for evaluating multiple checkpoint variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from neurovision_api.services.analytics import EvaluationReport


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
    model_a_name: str = Field(description="Name of first model")
    model_b_name: str = Field(description="Name of second model")
    model_a_version: str = Field(description="Version of first model")
    model_b_version: str = Field(description="Version of second model")
    compared_at: str = Field(description="ISO timestamp of comparison")
    metrics: list[ModelComparisonMetric] = Field(description="Per-metric comparisons")
    summary: str = Field(description="Human-readable summary of key differences")


class ModelComparisonStore:
    """Store and manage model comparisons."""

    def __init__(self, *, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.models: dict[str, ModelMetadata] = {}

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
        from datetime import datetime

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
        accuracy_improvement = "improved" if accuracy_diff > 0 else "declined"
        summary = (
            f"{name_b} {accuracy_improvement} accuracy by {abs(accuracy_pct):.2f}% "
            f"({report_b.accuracy:.4f} vs {report_a.accuracy:.4f}). "
            f"Loss {'decreased' if loss_diff < 0 else 'increased'} by {abs(loss_diff):.4f}."
        )

        return ModelComparison(
            model_a_name=name_a,
            model_b_name=name_b,
            model_a_version=report_a.model_version,
            model_b_version=report_b.model_version,
            compared_at=datetime.utcnow().isoformat() + "Z",
            metrics=metrics,
            summary=summary,
        )
