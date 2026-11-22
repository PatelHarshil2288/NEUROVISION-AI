from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from neurovision_api.api.v1.local_access import require_direct_local_client
from neurovision_api.services.inference import InferenceService
from neurovision_api.services.model_comparison import (
    AvailableCheckpoint,
    ModelActivation,
    ModelComparisonStore,
)

router = APIRouter(prefix="/model-registry", tags=["model-registry"])
logger = logging.getLogger(__name__)


class CheckpointSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ModelRegistrySnapshot(BaseModel):
    active: ModelActivation | None
    available_checkpoints: list[AvailableCheckpoint]
    history: list[ModelActivation]


def _snapshot(request: Request) -> ModelRegistrySnapshot:
    store: ModelComparisonStore = request.app.state.model_comparison_store
    service: InferenceService | None = request.app.state.inference_service
    try:
        checkpoints = store.list_checkpoints(
            loaded_checkpoint_hash=service.checkpoint_sha256 if service else None,
        )
        history = store.list_model_activations()
        persisted_active = store.active_model()
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        logger.error("Model registry could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail="The model registry is unavailable.") from exc

    active = None
    if service is not None and service.checkpoint_sha256:
        checkpoint = next(
            (item for item in checkpoints if item.checkpoint_hash == service.checkpoint_sha256),
            None,
        )
        if checkpoint is not None:
            activated_at = (
                persisted_active.activated_at
                if persisted_active and persisted_active.checkpoint_hash == checkpoint.checkpoint_hash
                else ""
            )
            active = ModelActivation(
                checkpoint_hash=checkpoint.checkpoint_hash,
                model_version=checkpoint.model_version,
                architecture=checkpoint.architecture,
                checkpoint_name=checkpoint.checkpoint_name,
                activated_at=activated_at,
            )
    return ModelRegistrySnapshot(
        active=active,
        available_checkpoints=checkpoints,
        history=history,
    )


def _activate(request: Request, checkpoint_hash: str) -> ModelRegistrySnapshot:
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        with request.app.state.model_registry_lock:
            checkpoint = store.checkpoint(checkpoint_hash)
            checkpoint_path = store.artifacts_dir / checkpoint.checkpoint_name
            service = InferenceService.from_checkpoint(checkpoint_path)
            if service.checkpoint_sha256 != checkpoint_hash:
                raise HTTPException(
                    status_code=409,
                    detail="The checkpoint changed while it was being loaded; activation was not applied.",
                )
            store.activate_model(checkpoint)
            request.app.state.inference_service = service
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except sqlite3.Error as exc:
        logger.error("Model activation could not be persisted: %s", exc)
        raise HTTPException(status_code=503, detail="The model registry is unavailable.") from exc
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.error("Checkpoint activation failed: %s", exc)
        raise HTTPException(status_code=409, detail="The selected checkpoint could not be activated.") from exc
    return _snapshot(request)


@router.get("", response_model=ModelRegistrySnapshot)
def model_registry(request: Request) -> ModelRegistrySnapshot:
    require_direct_local_client(request, "Model registry")
    return _snapshot(request)


@router.post("/activate", response_model=ModelRegistrySnapshot)
def activate_model(payload: CheckpointSelection, request: Request) -> ModelRegistrySnapshot:
    require_direct_local_client(request, "Model activation")
    return _activate(request, payload.checkpoint_hash)


@router.post("/rollback", response_model=ModelRegistrySnapshot)
def rollback_model(request: Request) -> ModelRegistrySnapshot:
    require_direct_local_client(request, "Model rollback")
    store: ModelComparisonStore = request.app.state.model_comparison_store
    try:
        previous = store.previous_model_activation()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        logger.error("Model rollback history could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail="The model registry is unavailable.") from exc
    return _activate(request, previous.checkpoint_hash)
