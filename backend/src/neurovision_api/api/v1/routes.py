from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from neurovision_api.api.v1.schemas import (
    ActivationsResponse,
    ImageRequest,
    ModelInfo,
    ModelListResponse,
    ModelWeightsResponse,
    PredictionResponse,
    SaliencyRequest,
    SaliencyResponse,
)
from neurovision_api.services.inference import InferenceService

router = APIRouter()
MODEL_ID = "mnist-mlp-v1"


def _inference_service(request: Request) -> InferenceService:
    service = request.app.state.inference_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained MNIST checkpoint is available.",
        )
    return service


def _architecture_dimensions(service: InferenceService | None) -> list[int]:
    if service is None:
        return [784, 128, 64, 10]
    if service.architecture == "mlp":
        return [
            service.model_config["input_dim"],
            service.model_config["hidden1"],
            service.model_config["hidden2"],
            service.model_config["output_dim"],
        ]
    return [1, 28, 28, 10]


def _model_id(service: InferenceService | None) -> str:
    if service is None or service.architecture == "mlp":
        return MODEL_ID
    return f"mnist-{service.architecture}-v1"


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_ready": request.app.state.inference_service is not None,
    }


@router.get("/models", response_model=ModelListResponse)
def models(request: Request) -> ModelListResponse:
    service: InferenceService | None = request.app.state.inference_service
    return ModelListResponse(
        models=[
            ModelInfo(
                model_id=_model_id(service),
                model_version=service.model_version if service else None,
                architecture=_architecture_dimensions(service),
                parameter_count=service.parameter_count if service else 109_386,
                available=service is not None,
            )
        ]
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: ImageRequest, request: Request) -> PredictionResponse:
    service = _inference_service(request)
    result = service.predict(payload.pixels)
    return PredictionResponse(model_id=_model_id(service), **result)


@router.post("/saliency", response_model=SaliencyResponse)
def saliency(payload: SaliencyRequest, request: Request) -> SaliencyResponse:
    service = _inference_service(request)
    result = service.saliency(payload.pixels, payload.target_class)
    return SaliencyResponse(model_id=_model_id(service), **result)


@router.post("/activations", response_model=ActivationsResponse)
def activations(
    payload: ImageRequest,
    request: Request,
) -> ActivationsResponse:
    service = _inference_service(request)
    try:
        result = service.activations(payload.pixels)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return ActivationsResponse(model_id=_model_id(service), **result)


@router.get("/weights", response_model=ModelWeightsResponse)
def weights(request: Request) -> ModelWeightsResponse:
    service = _inference_service(request)
    try:
        layers = service.weights()
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return ModelWeightsResponse(
        model_id=_model_id(service),
        model_version=service.model_version,
        architecture=_architecture_dimensions(service),
        layers=layers,
    )
