from __future__ import annotations

from pathlib import Path
import os
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neurovision_api import __version__
from neurovision_api.api.v1.routes import router as v1_router
from neurovision_api.api.v1.training import router as training_router
from neurovision_api.api.v1.analytics import router as analytics_router
from neurovision_api.api.v1.comparison import router as comparison_router
from neurovision_api.api.v1.model_registry import router as model_registry_router
from neurovision_api.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR, get_allowed_origins, get_checkpoint_path
from neurovision_api.services.inference import InferenceService
from neurovision_api.services.training_lab import TrainingLab
from neurovision_api.services.analytics import AnalyticsStore
from neurovision_api.services.model_comparison import ModelComparisonStore


def create_app(
    *,
    service: InferenceService | None = None,
    checkpoint_path: str | Path | None = None,
    analytics_report_path: str | Path | None = None,
    analytics_data_dir: str | Path | None = None,
    comparison_artifacts_dir: str | Path | None = None,
) -> FastAPI:
    artifacts_dir = (
        Path(comparison_artifacts_dir)
        if comparison_artifacts_dir is not None
        else DEFAULT_ARTIFACTS_DIR
    )
    comparison_store = ModelComparisonStore(artifacts_dir=artifacts_dir)
    comparison_store.claim_model_registry_worker()
    resolved_checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else get_checkpoint_path()
    )
    checkpoint_is_explicit = (
        checkpoint_path is not None
        or bool(os.environ.get("NEUROVISION_MNIST_CHECKPOINT"))
    )
    injected_service = service is not None
    if service is None and not checkpoint_is_explicit:
        active_model = comparison_store.active_model()
        if active_model is not None:
            active_path = comparison_store.checkpoint_path(active_model.checkpoint_hash)
            service = InferenceService.from_checkpoint(active_path)
            if service.checkpoint_sha256 != active_model.checkpoint_hash:
                raise RuntimeError("The active checkpoint changed while it was being loaded.")
    if service is None and resolved_checkpoint.is_file():
        service = InferenceService.from_checkpoint(resolved_checkpoint)
    if (
        service is not None
        and not injected_service
        and not checkpoint_is_explicit
        and comparison_store.active_model() is None
        and service.checkpoint_sha256
    ):
        for checkpoint in comparison_store.list_checkpoints():
            if checkpoint.checkpoint_hash == service.checkpoint_sha256:
                comparison_store.initialize_active_model(checkpoint)
                break

    application = FastAPI(
        title="NEUROVISION AI API",
        version=__version__,
        description=(
            "Versioned inference, activation, and model-parameter API "
            "for the web platform."
        ),
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_response(
        request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        errors = [
            {
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"],
            }
            for error in exception.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    application.state.inference_service = service
    application.state.analytics_store = AnalyticsStore(
        report_path=Path(analytics_report_path) if analytics_report_path is not None else DEFAULT_ARTIFACTS_DIR / "analytics-report.json",
        data_dir=Path(analytics_data_dir) if analytics_data_dir is not None else DEFAULT_DATA_DIR,
    )
    application.state.model_comparison_store = comparison_store
    application.state.model_registry_lock = threading.RLock()
    application.state.training_lab = TrainingLab(
        enabled=os.environ.get("NEUROVISION_TRAINING_ENABLED") == "1",
        data_dir=DEFAULT_DATA_DIR,
        checkpoint_dir=DEFAULT_ARTIFACTS_DIR / "training-lab",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(v1_router, prefix="/api/v1")
    application.include_router(training_router, prefix="/api/v1")
    application.include_router(analytics_router, prefix="/api/v1")
    application.include_router(comparison_router, prefix="/api/v1")
    application.include_router(model_registry_router, prefix="/api/v1")
    return application


app = create_app()
