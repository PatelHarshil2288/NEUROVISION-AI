from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from neurovision_api.api.v1.local_access import require_direct_local_client
from neurovision_api.services.training_lab import TrainingLab

router = APIRouter(prefix="/training", tags=["local training lab"])


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=1, ge=1, le=3, strict=True)


def _local_lab(request: Request) -> TrainingLab:
    require_direct_local_client(request, "Training")
    return request.app.state.training_lab


@router.get("")
def training_status(request: Request) -> dict:
    return _local_lab(request).snapshot()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def start_training(payload: TrainingRequest, request: Request) -> dict:
    lab = _local_lab(request)
    try:
        return lab.start(payload.epochs)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
