"""Integration tests for model comparison API endpoints."""

import pytest
from fastapi.testclient import TestClient

from neurovision_api.main import create_app
from neurovision_api.services.inference import InferenceService
from neurovision_api.services.model_comparison import ModelComparisonStore


@pytest.fixture
def test_app():
    """Create test application."""
    return create_app(service=None)


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


def test_comparison_summary_endpoint(client):
    """Test model comparison summary endpoint."""
    response = client.get("/api/v1/comparison/summary")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_models" in data
    assert "models_registered" in data
    assert "feature_status" in data
    assert "message" in data
    
    # Check feature status structure
    features = data["feature_status"]
    assert "multi_model_evaluation" in features
    assert "historical_comparison" in features
    assert "version_tracking" in features


def test_available_models_endpoint(client):
    """Test listing available models endpoint."""
    response = client.get("/api/v1/comparison/available-models")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)
    assert data["count"] >= 0


def test_model_comparison_requires_multiple_models(client):
    """Test that model comparison handles missing reports gracefully."""
    response = client.post(
        "/api/v1/comparison/reports",
        json={
            "model_a_checkpoint_hash": "a" * 64,
            "model_b_checkpoint_hash": "b" * 64,
        }
    )
    
    # Should fail with 503 (unavailable) since no reports loaded
    assert response.status_code in [503, 501]


def test_comparison_endpoints_dont_break_existing_api(client):
    """Ensure new comparison endpoints don't interfere with existing API."""
    # Test that existing health endpoint still works
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    # Test that existing model endpoint still works
    response = client.get("/api/v1/models")
    assert response.status_code == 200
