"""Model comparison endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from neurovision_api.api.v1.routes import _inference_service
from neurovision_api.services.analytics import AnalyticsStore
from neurovision_api.services.model_comparison import ModelComparison, ModelComparisonStore

router = APIRouter(prefix="/comparison", tags=["model-comparison"])
logger = logging.getLogger(__name__)


class ComparisonRequest(BaseModel):
    """Request to compare two model reports."""
    model_a_checkpoint_hash: str = Field(description="SHA256 hash of model A checkpoint")
    model_b_checkpoint_hash: str = Field(description="SHA256 hash of model B checkpoint")
    model_a_name: str = Field(default="Model A", description="Display name for model A")
    model_b_name: str = Field(default="Model B", description="Display name for model B")


@router.get("/available-models", response_model=dict)
def available_models(request: Request) -> dict:
    """List available models for comparison."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    models = store.list_models()
    return {
        "models": [
            {
                "name": m.name,
                "version": m.version,
                "checkpoint_hash": m.compute_hash(),
                "is_loaded": m.is_loaded,
            }
            for m in models
        ],
        "count": len(models),
    }


@router.post("/reports", response_model=ModelComparison)
def compare_models(request_body: ComparisonRequest, request: Request) -> ModelComparison:
    """Compare two model evaluation reports."""
    analytics_store: AnalyticsStore = request.app.state.analytics_store
    
    # For now, we'll compare the loaded model against historical reports
    # In a full implementation, this would load multiple checkpoint reports
    try:
        current_report = analytics_store.report(_inference_service(request))
    except (OSError, ValueError, ValidationError) as exc:
        logger.warning("Current model report unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=f"Current model report unavailable: {exc}") from exc

    # This is a placeholder - in production you'd load the comparison report
    raise HTTPException(
        status_code=501,
        detail="Model comparison requires multiple stored reports. "
               "Generate and store evaluation reports for different checkpoints first.",
    )


@router.get("/summary", response_model=dict)
def comparison_summary(request: Request) -> dict:
    """Get a summary of model capabilities for comparison."""
    store: ModelComparisonStore = request.app.state.model_comparison_store
    models = store.list_models()
    
    return {
        "total_models": len(models),
        "models_registered": len(models) > 0,
        "feature_status": {
            "multi_model_evaluation": len(models) >= 2,
            "historical_comparison": len(models) > 1,
            "version_tracking": True,
        },
        "message": "Model comparison is available when multiple evaluation reports are present.",
    }
