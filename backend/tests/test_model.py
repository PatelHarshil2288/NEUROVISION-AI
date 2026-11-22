import pytest
import torch

from neurovision_api.ml.model import MLP
from neurovision_api.services.inference import InferenceService


def test_mlp_forward_shapes_and_parameter_count() -> None:
    model = MLP()
    result = model.forward_with_activations(torch.zeros(2, 1, 28, 28))

    assert model.parameter_count == 109_386
    assert result["logits"].shape == (2, 10)
    assert result["a1"].shape == (2, 128)
    assert result["a2"].shape == (2, 64)
    assert torch.equal(model(torch.zeros(2, 784)), result["logits"])


def test_inference_service_exposes_exact_layer_parameters() -> None:
    model = MLP()
    service = InferenceService(model=model, model_version="test-mnist-v1")

    weights = service.weights()

    torch.testing.assert_close(
        torch.tensor(weights["input_to_hidden_1"]["weights"]),
        model.fc1.weight,
    )
    torch.testing.assert_close(
        torch.tensor(weights["hidden_1_to_hidden_2"]["biases"]),
        model.fc2.bias,
    )


@pytest.mark.parametrize(
    "invalid_input",
    [
        torch.zeros(1, 27, 28),
        torch.zeros(1, 1, 27, 28),
        torch.zeros(1, 785),
    ],
)
def test_mlp_rejects_invalid_input_shape(invalid_input: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        MLP()(invalid_input)
