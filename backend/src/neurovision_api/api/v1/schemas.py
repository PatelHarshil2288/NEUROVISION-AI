from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pixels: list[list[float]]

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, pixels: list[list[float]]) -> list[list[float]]:
        if len(pixels) != 28 or any(len(row) != 28 for row in pixels):
            raise ValueError("pixels must be a 28x28 grayscale image.")
        for row in pixels:
            for value in row:
                if not math.isfinite(value):
                    raise ValueError("Pixel values must be finite.")
                if value < 0.0 or value > 1.0:
                    raise ValueError("Pixel values must be in the range [0, 1].")
        return pixels


class PredictionResponse(BaseModel):
    model_id: str
    model_version: str
    predicted_class: int = Field(ge=0, le=9)
    logits: list[float]
    probabilities: list[float]
    inference_ms: float = Field(ge=0)


class SaliencyRequest(ImageRequest):
    target_class: int | None = Field(default=None, ge=0, le=9, strict=True)


class SaliencyResponse(BaseModel):
    model_id: str
    model_version: str
    predicted_class: int
    target_class: int
    probabilities: list[float]
    gradient: list[list[float]]


class LayerActivation(BaseModel):
    pre_activations: list[float]
    activations: list[float]


class ActivationsResponse(BaseModel):
    model_id: str
    model_version: str
    predicted_class: int = Field(ge=0, le=9)
    logits: list[float]
    probabilities: list[float]
    layers: dict[str, LayerActivation]


class LayerWeights(BaseModel):
    weights: list[list[float]]
    biases: list[float]


class ModelWeightsResponse(BaseModel):
    model_id: str
    model_version: str
    architecture: list[int]
    layers: dict[str, LayerWeights]


class ModelInfo(BaseModel):
    model_id: str
    model_version: str | None
    architecture: list[int]
    parameter_count: int
    available: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
