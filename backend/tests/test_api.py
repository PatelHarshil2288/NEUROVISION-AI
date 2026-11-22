import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from neurovision_api.main import create_app
from neurovision_api.ml.checkpoints import save_checkpoint
from neurovision_api.ml.model import MLP
from neurovision_api.services.inference import InferenceService


@pytest.fixture
def client(tmp_path) -> TestClient:
    model = MLP()
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "epoch": 1,
            "metadata": {"model_version": "test-mnist-v1"},
        },
    )
    service = InferenceService.from_checkpoint(checkpoint_path)
    return TestClient(create_app(service=service))


def valid_image() -> dict[str, list[list[float]]]:
    return {"pixels": [[0.0 for _ in range(28)] for _ in range(28)]}


def test_prediction_returns_model_outputs_and_normalized_probabilities(
    client: TestClient,
) -> None:
    response = client.post("/api/v1/predict", json=valid_image())

    assert response.status_code == 200
    result = response.json()
    assert result["model_id"] == "mnist-mlp-v1"
    assert result["model_version"] == "test-mnist-v1"
    assert 0 <= result["predicted_class"] <= 9
    assert len(result["logits"]) == 10
    assert sum(result["probabilities"]) == pytest.approx(1.0)
    assert result["inference_ms"] >= 0


def test_activations_endpoint_returns_real_layer_values(client: TestClient) -> None:
    response = client.post("/api/v1/activations", json=valid_image())

    assert response.status_code == 200
    result = response.json()
    assert len(result["layers"]["input"]["activations"]) == 784
    assert result["layers"]["input"]["activations"][0] == pytest.approx(
        (0.0 - 0.1307) / 0.3081,
    )
    assert len(result["layers"]["hidden_1"]["pre_activations"]) == 128
    assert len(result["layers"]["hidden_2"]["activations"]) == 64
    assert len(result["layers"]["output"]["activations"]) == 10


def test_weights_endpoint_returns_model_version_and_layer_shapes(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/weights")

    assert response.status_code == 200
    result = response.json()
    assert result["model_id"] == "mnist-mlp-v1"
    assert result["model_version"] == "test-mnist-v1"
    assert result["architecture"] == [784, 128, 64, 10]
    assert len(result["layers"]["input_to_hidden_1"]["weights"]) == 128
    assert len(result["layers"]["input_to_hidden_1"]["weights"][0]) == 784
    assert len(result["layers"]["input_to_hidden_1"]["biases"]) == 128
    assert len(result["layers"]["hidden_1_to_hidden_2"]["weights"]) == 64
    assert len(result["layers"]["hidden_1_to_hidden_2"]["weights"][0]) == 128
    assert len(result["layers"]["hidden_1_to_hidden_2"]["biases"]) == 64
    assert len(result["layers"]["hidden_2_to_output"]["weights"]) == 10
    assert len(result["layers"]["hidden_2_to_output"]["weights"][0]) == 64
    assert len(result["layers"]["hidden_2_to_output"]["biases"]) == 10


@pytest.mark.parametrize(
    "pixels",
    [
        [[0.0 for _ in range(27)] for _ in range(28)],
        [[-0.1 for _ in range(28)] for _ in range(28)],
        [[float("nan") for _ in range(28)] for _ in range(28)],
    ],
)
def test_api_rejects_invalid_images(client: TestClient, pixels) -> None:
    response = client.post(
        "/api/v1/predict",
        content=json.dumps({"pixels": pixels}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_missing_checkpoint_reports_not_ready_and_does_not_predict(tmp_path) -> None:
    client = TestClient(create_app(checkpoint_path=tmp_path / "missing.pt"))

    health = client.get("/api/v1/health")
    models = client.get("/api/v1/models")
    prediction = client.post("/api/v1/predict", json=valid_image())
    weights = client.get("/api/v1/weights")

    assert health.status_code == 200
    assert health.json()["model_ready"] is False
    assert models.json()["models"][0]["available"] is False
    assert prediction.status_code == 503
    assert weights.status_code == 503


def test_cors_allows_both_local_frontend_origins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("NEUROVISION_ALLOWED_ORIGINS", raising=False)
    client = TestClient(create_app(checkpoint_path=tmp_path / "missing.pt"))

    for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        response = client.options(
            "/api/v1/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
