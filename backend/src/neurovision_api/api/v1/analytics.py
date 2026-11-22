from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from neurovision_api.api.v1.local_access import require_direct_local_client
from neurovision_api.api.v1.routes import _inference_service
from neurovision_api.services.analytics import EvaluationReport, test_dataset_hashes
from neurovision_api.services.evaluation import evaluate_mnist_checkpoint
from neurovision_api.services.model_comparison import ModelComparisonStore

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)


def _report(request: Request) -> EvaluationReport:
    require_direct_local_client(request, "Evaluation")
    service = _inference_service(request)
    comparison_store: ModelComparisonStore = request.app.state.model_comparison_store
    if not service.checkpoint_sha256:
        raise HTTPException(status_code=503, detail="The active model has no checkpoint identity.")
    try:
        checkpoint_path = comparison_store.checkpoint_path(service.checkpoint_sha256)
        try:
            report = comparison_store.get_report(service.checkpoint_sha256)
        except FileNotFoundError:
            report = comparison_store.save_report(
                evaluate_mnist_checkpoint(
                    checkpoint_path,
                    request.app.state.analytics_store.data_dir,
                )
            )
        current_hashes = test_dataset_hashes(request.app.state.analytics_store.data_dir)
        if (report.dataset_images_sha256, report.dataset_labels_sha256) != current_hashes:
            raise HTTPException(
                status_code=409,
                detail="The saved evaluation uses different MNIST test data.",
            )
        if report.model_version != service.model_version:
            raise HTTPException(
                status_code=409,
                detail="The saved evaluation does not match the active model version.",
            )
        return report
    except HTTPException:
        raise
    except (OSError, RuntimeError, sqlite3.Error, ValueError, ValidationError) as exc:
        logger.warning("Evaluation report unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/evaluation", response_model=EvaluationReport)
def evaluation(request: Request) -> EvaluationReport:
    return _report(request)


@router.get("/misclassifications/{sample_index}")
def misclassification_image(sample_index: int, request: Request) -> dict:
    report = _report(request)
    store: AnalyticsStore = request.app.state.analytics_store
    try:
        pixels = store.error_image(report, sample_index)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Misclassification image unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"sample_index": sample_index, "checkpoint_sha256": report.checkpoint_sha256, "pixels": pixels}
