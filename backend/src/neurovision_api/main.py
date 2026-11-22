from __future__ import annotations

from pathlib import Path
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neurovision_api import __version__
from neurovision_api.api.v1.routes import router as v1_router
from neurovision_api.api.v1.training import router as training_router
from neurovision_api.api.v1.analytics import router as analytics_router
from neurovision_api.api.v1.comparison import router as comparison_router
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
) -> FastAPI:
    resolved_checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else get_checkpoint_path()
    )
    if service is None and resolved_checkpoint.is_file():
        service = InferenceService.from_checkpoint(resolved_checkpoint)

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
    application.state.model_comparison_store = ModelComparisonStore(
        artifacts_dir=DEFAULT_ARTIFACTS_DIR,
    )
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
    return application


app = create_app()
