"""Model comparison endpoints."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from neurovision_api.api.v1.local_access import require_direct_local_client
from neurovision_api.services.analytics import EvaluationReport, test_dataset_hashes
from neurovision_api.services.evaluation import evaluate_mnist_checkpoint
from neurovision_api.services.model_comparison import (
    AvailableCheckpoint,
    ModelComparison,
    ModelComparisonStore,
)

router = APIRouter(prefix="/comparison", tags=["model-comparison"])
logger = logging.getLogger(__name__)


class ComparisonRequest(BaseModel):
    """Request to compare two model reports."""
    model_config = ConfigDict(extra="forbid")
    model_a_checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$", description="SHA256 hash of model A checkpoint")
    model_b_checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$", description="SHA256 hash of model B checkpoint")
    model_a_name: str = Field(default="Model A", min_length=1, max_length=100, description="Display name for model A")
    model_b_name: str = Field(default="Model B", min_length=1, max_length=100, description="Display name for model B")


class CheckpointEvaluationRequest(BaseModel):
    """Request a server-side evaluation of an approved local checkpoint."""
    model_config = ConfigDict(extra="forbid")
    checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


@router.get("/checkpoints", response_model=list[AvailableCheckpoint])
def available_checkpoints(request: Request) -> list[AvailableCheckpoint]:
    """List checkpoints from the approved local artifact directories."""
    require_direct_local_client(request, "Checkpoint discovery")
    store: ModelComparisonStore = request.app.state.model_comparison_store
    active_service = request.app.state.inference_service
    try:
        return store.list_checkpoints(
            loaded_checkpoint_hash=active_service.checkpoint_sha256 if active_service else None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Available checkpoints could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/available-models", response_model=dict)
def available_models(request: Request) -> dict:
    """List available models for comparison."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        reports = store.list_reports()
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.error("Available evaluation reports could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    active_service = request.app.state.inference_service
    return {
        "models": [
            {
                "name": report.model_version,
                "version": report.model_version,
                "checkpoint_hash": report.checkpoint_sha256,
                "is_loaded": (
                    active_service is not None
                    and active_service.checkpoint_sha256 == report.checkpoint_sha256
                ),
            }
            for report in reports
        ],
        "count": len(reports),
    }


@router.get("/evaluation-reports", response_model=list[EvaluationReport])
def evaluation_reports(request: Request) -> list[EvaluationReport]:
    """List saved evaluation reports for model comparison."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        return store.list_reports()
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.error("Stored evaluation reports could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/evaluation-reports",
    response_model=EvaluationReport,
    status_code=status.HTTP_201_CREATED,
)
def save_evaluation_report(
    payload: CheckpointEvaluationRequest,
    request: Request,
) -> EvaluationReport:
    """Evaluate an approved checkpoint against the server's official MNIST test data."""
    require_direct_local_client(request, "Evaluation")
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        checkpoint_path = store.checkpoint_path(payload.checkpoint_hash)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Approved checkpoint could not be resolved: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    data_dir = request.app.state.analytics_store.data_dir
    try:
        try:
            report = store.get_report(payload.checkpoint_hash)
        except FileNotFoundError:
            report = None
        if report is not None:
            current_hashes = test_dataset_hashes(data_dir)
            if (report.dataset_images_sha256, report.dataset_labels_sha256) != current_hashes:
                raise HTTPException(
                    status_code=409,
                    detail="The stored report uses different MNIST test data; create a new checkpoint before evaluating changed data.",
                )
            return report
        report = evaluate_mnist_checkpoint(checkpoint_path, data_dir)
        return store.save_report(report)
    except HTTPException:
        raise
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        logger.error("Checkpoint evaluation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/reports", response_model=ModelComparison)
def compare_models(request_body: ComparisonRequest, request: Request) -> ModelComparison:
    """Compare two persisted evaluation reports and save the result."""
    require_direct_local_client(request, "Comparison")
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        report_a = store.get_report(request_body.model_a_checkpoint_hash)
        report_b = store.get_report(request_body.model_b_checkpoint_hash)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, sqlite3.Error, ValueError, ValidationError) as exc:
        logger.warning("Evaluation reports could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        comparison = store.compare_reports(
            report_a,
            report_b,
            name_a=request_body.model_a_name,
            name_b=request_body.model_b_name,
        )
        return store.save_comparison(comparison)
    except sqlite3.Error as exc:
        logger.error("Comparison could not be stored: %s", exc)
        raise HTTPException(status_code=503, detail="The comparison database is unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/reports", response_model=list[ModelComparison])
def comparison_history(request: Request) -> list[ModelComparison]:
    """List saved comparison results, newest first."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        return store.list_comparisons()
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.error("Saved comparisons could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/summary", response_model=dict)
def comparison_summary(request: Request) -> dict:
    """Get a summary of model capabilities for comparison."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        reports = store.list_reports()
        comparisons = store.list_comparisons()
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.error("Comparison summary could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "total_models": len(reports),
        "models_registered": len(reports) > 0,
        "total_comparisons": len(comparisons),
        "feature_status": {
            "multi_model_evaluation": len(reports) >= 2,
            "historical_comparison": len(comparisons) > 0,
            "version_tracking": True,
        },
        "message": "Evaluation reports and comparison results are stored locally.",
    }
