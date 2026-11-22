from __future__ import annotations

from pathlib import Path
from io import BytesIO
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from neurovision_api.ml.checkpoints import load_checkpoint
from neurovision_api.ml.model_factory import create_model, describe_model
from neurovision_api.ml.model import MLP

class InferenceService:
    def __init__(
        self,
        *,
        model: nn.Module,
        model_version: str,
        device: torch.device | None = None,
        checkpoint_sha256: str | None = None,
        architecture: str = "mlp",
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.model_version = model_version
        self.checkpoint_sha256 = checkpoint_sha256
        self.architecture = architecture
        self.model_config = model_config or describe_model(model)[1]
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> InferenceService:
        target_device = device or torch.device("cpu")
        checkpoint_bytes = Path(path).read_bytes()
        checkpoint = load_checkpoint(BytesIO(checkpoint_bytes), map_location=target_device)
        architecture = checkpoint.get("architecture", "mlp")
        model_config = checkpoint["model_config"]
        model = create_model(architecture, model_config)
        model.load_state_dict(checkpoint["model_state"])
        model_version = checkpoint["metadata"]["model_version"]
        return cls(
            model=model,
            model_version=model_version,
            device=target_device,
            checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            architecture=architecture,
            model_config=model_config,
        )

    def predict(self, pixels: list[list[float]]) -> dict[str, Any]:
        image = self._prepare_image(pixels)
        started = perf_counter()
        with torch.inference_mode():
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
        elapsed_ms = (perf_counter() - started) * 1000.0
        probability_values = probabilities[0].cpu().tolist()
        return {
            "model_version": self.model_version,
            "predicted_class": int(probabilities.argmax(dim=1).item()),
            "logits": logits[0].cpu().tolist(),
            "probabilities": probability_values,
            "inference_ms": elapsed_ms,
        }

    def activations(self, pixels: list[list[float]]) -> dict[str, Any]:
        if not isinstance(self.model, MLP):
            raise NotImplementedError("Detailed activation visualization is currently available only for MLP checkpoints.")
        image = self._prepare_image(pixels)
        with torch.inference_mode():
            outputs = self.model.forward_with_activations(image)
            probabilities = torch.softmax(outputs["logits"], dim=1)
        return {
            "model_version": self.model_version,
            "predicted_class": int(probabilities.argmax(dim=1).item()),
            "logits": outputs["logits"][0].cpu().tolist(),
            "probabilities": probabilities[0].cpu().tolist(),
            "layers": {
                "input": {
                    "pre_activations": [],
                    "activations": outputs["input"][0].cpu().tolist(),
                },
                "hidden_1": {
                    "pre_activations": outputs["z1"][0].cpu().tolist(),
                    "activations": outputs["a1"][0].cpu().tolist(),
                },
                "hidden_2": {
                    "pre_activations": outputs["z2"][0].cpu().tolist(),
                    "activations": outputs["a2"][0].cpu().tolist(),
                },
                "output": {
                    "pre_activations": outputs["logits"][0].cpu().tolist(),
                    "activations": probabilities[0].cpu().tolist(),
                },
            },
        }

    def weights(self) -> dict[str, dict[str, Any]]:
        if not isinstance(self.model, MLP):
            raise NotImplementedError("Weight visualization is currently available only for MLP checkpoints.")
        return {
            "input_to_hidden_1": {
                "weights": self.model.fc1.weight.detach().cpu().tolist(),
                "biases": self.model.fc1.bias.detach().cpu().tolist(),
            },
            "hidden_1_to_hidden_2": {
                "weights": self.model.fc2.weight.detach().cpu().tolist(),
                "biases": self.model.fc2.bias.detach().cpu().tolist(),
            },
            "hidden_2_to_output": {
                "weights": self.model.fc3.weight.detach().cpu().tolist(),
                "biases": self.model.fc3.bias.detach().cpu().tolist(),
            },
        }

    def saliency(self, pixels: list[list[float]], target_class: int | None = None) -> dict[str, Any]:
        # Validate using the same input contract as prediction before retaining gradients.
        self._prepare_image(pixels)
        raw = torch.tensor(pixels, dtype=torch.float32, device=self.device).reshape(1, 1, 28, 28)
        raw.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model((raw - 0.1307) / 0.3081)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(probabilities.argmax(dim=1).item())
            selected_class = predicted_class if target_class is None else target_class
            gradient = torch.autograd.grad(logits[0, selected_class], raw)[0]
        return {
            "model_version": self.model_version,
            "predicted_class": predicted_class,
            "target_class": selected_class,
            "probabilities": probabilities[0].detach().cpu().tolist(),
            "gradient": gradient.reshape(28, 28).detach().cpu().tolist(),
        }

    def _prepare_image(self, pixels: list[list[float]]) -> torch.Tensor:
        image = np.asarray(pixels, dtype=np.float32)
        if image.shape != (28, 28):
            raise ValueError(f"Image must have shape (28, 28); got {image.shape}.")
        if not np.isfinite(image).all():
            raise ValueError("Image values must be finite.")
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("Image values must be in the range [0, 1].")

        normalized = (image - 0.1307) / 0.3081
        return torch.from_numpy(normalized).reshape(1, 1, 28, 28).to(self.device)
