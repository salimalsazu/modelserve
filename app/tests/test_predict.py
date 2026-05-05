"""
Test suite for prediction endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock trained model."""
    model = MagicMock()
    model.predict.return_value = [1.0]
    model.predict_proba.return_value = [[0.3, 0.7]]
    return model


@pytest.fixture
def mock_features():
    """Mock feature store response."""
    return {
        "entity_id": 1,
        "feature_1": 0.5,
        "feature_2": 0.3,
        "feature_3": 0.8,
        "feature_4": 0.2,
        "feature_5": 0.6,
        "feature_6": 0.4,
        "feature_7": 0.9,
        "feature_8": 0.1,
        "feature_9": 0.7,
        "feature_10": 0.3,
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data

    def test_health_components(self, client):
        """Health should report all component statuses."""
        response = client.get("/health")
        data = response.json()
        
        assert "api" in data["components"]
        assert "model" in data["components"]
        assert "feature_store" in data["components"]


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_200(self, client, mock_model, mock_features):
        """Predict endpoint should return 200 OK with valid input."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                response = client.post(
                    "/predict",
                    json={"entity_id": 1, "features": [0.5] * 10}
                )
        
        assert response.status_code == 200

    def test_predict_response_structure(self, client, mock_model, mock_features):
        """Predict response should have required fields."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                response = client.post(
                    "/predict",
                    json={"entity_id": 1, "features": [0.5] * 10}
                )
        
        data = response.json()
        
        assert "prediction" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_predict_without_entity_id(self, client, mock_model):
        """Predict should work with only features (no entity_id)."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            response = client.post(
                "/predict",
                json={"features": [0.5] * 10}
            )
        
        assert response.status_code == 200

    def test_predict_invalid_features_type(self, client, mock_model):
        """Predict should reject invalid features type."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            response = client.post(
                "/predict",
                json={"entity_id": 1, "features": "invalid"}
            )
        
        assert response.status_code == 422  # Validation error

    def test_predict_missing_features(self, client, mock_model):
        """Predict should require features field."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            response = client.post(
                "/predict",
                json={"entity_id": 1}
            )
        
        assert response.status_code == 422

    def test_predict_wrong_feature_count(self, client, mock_model):
        """Predict should handle feature count mismatch gracefully."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            response = client.post(
                "/predict",
                json={"entity_id": 1, "features": [0.5]}  # Only 1 feature
            )
        
        # Should still return 200 with model handling the mismatch
        assert response.status_code in [200, 422]


class TestPredictEntityEndpoint:
    """Tests for /predict/{entity_id} endpoint."""

    def test_predict_entity_returns_200(self, client, mock_model, mock_features):
        """Predict with entity_id should return 200 OK."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                response = client.get("/predict/1")
        
        assert response.status_code == 200

    def test_predict_entity_response_structure(self, client, mock_model, mock_features):
        """Predict entity response should include features."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                response = client.get("/predict/1")
        
        data = response.json()
        
        assert "prediction" in data
        assert "entity_id" in data
        assert "features" in data

    def test_predict_entity_with_explain(self, client, mock_model, mock_features):
        """Predict with explain=true should include SHAP values."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                response = client.get("/predict/1?explain=true")
        
        assert response.status_code == 200
        data = response.json()
        assert "shap_values" in data

    def test_predict_entity_not_found(self, client, mock_model):
        """Predict for non-existent entity should return mock features."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=None):
                response = client.get("/predict/999999")
        
        assert response.status_code == 200
        # Should still return prediction with mock features
        data = response.json()
        assert "prediction" in data


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Metrics should return Prometheus format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_prediction_metrics(self, client):
        """Metrics should contain prediction-related metrics."""
        response = client.get("/metrics")
        content = response.text
        
        assert "prediction_requests_total" in content or "prediction_duration" in content


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, client, mock_model):
        """Model info endpoint should return 200 OK."""
        with patch("app.main.model_loader.get_model_info", return_value={
            "name": "test_model",
            "version": "1",
            "stage": "Production",
            "uri": "s3://bucket/model"
        }):
            response = client.get("/model/info")
        
        assert response.status_code == 200

    def test_model_info_structure(self, client, mock_model):
        """Model info should contain version and stage."""
        with patch("app.main.model_loader.get_model_info", return_value={
            "name": "test_model",
            "version": "1",
            "stage": "Production",
            "uri": "s3://bucket/model"
        }):
            response = client.get("/model/info")
        
        data = response.json()
        assert "version" in data
        assert "stage" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_load_failure(self, client):
        """Should handle model loading failures gracefully."""
        with patch("app.main.model_loader.load_model", side_effect=Exception("Model not found")):
            response = client.post(
                "/predict",
                json={"entity_id": 1, "features": [0.5] * 10}
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_feature_store_failure(self, client, mock_model):
        """Should handle feature store failures with fallback."""
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", side_effect=Exception("Redis down")):
                response = client.post(
                    "/predict",
                    json={"entity_id": 1, "features": [0.5] * 10}
                )
        
        # Should still work with fallback features
        assert response.status_code == 200

    def test_invalid_json_body(self, client):
        """Should handle invalid JSON body."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_empty_body(self, client):
        """Should handle empty request body."""
        response = client.post(
            "/predict",
            content="",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code in [422, 500]


class TestPerformance:
    """Tests for performance requirements."""

    def test_prediction_latency_reasonable(self, client, mock_model, mock_features):
        """Predictions should complete within reasonable time."""
        import time
        
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                start = time.time()
                response = client.post(
                    "/predict",
                    json={"entity_id": 1, "features": [0.5] * 10}
                )
                latency = time.time() - start
        
        assert response.status_code == 200
        # Should complete in under 1 second
        assert latency < 1.0

    def test_concurrent_requests(self, client, mock_model, mock_features):
        """Should handle concurrent requests."""
        import concurrent.futures
        
        with patch("app.main.model_loader.load_model", return_value=mock_model):
            with patch("app.main.feature_client.get_online_features", return_value=mock_features):
                def make_request():
                    return client.post(
                        "/predict",
                        json={"entity_id": 1, "features": [0.5] * 10}
                    )
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(make_request) for _ in range(10)]
                    responses = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)


class TestIntegration:
    """Integration tests (require running services)."""

    @pytest.fixture
    def integration_client(self):
        """Create client for integration testing."""
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_full_prediction_flow(self, integration_client):
        """Test complete prediction flow with health check."""
        # Check health first
        health = integration_client.get("/health")
        if health.status_code != 200:
            pytest.skip("Services not running")
        
        # Make prediction
        response = integration_client.post(
            "/predict",
            json={"entity_id": 1, "features": [0.5] * 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "prediction" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert isinstance(data["prediction"], (int, float, list))

    def test_metrics_endpoint_format(self, integration_client):
        """Verify metrics endpoint returns Prometheus format."""
        response = integration_client.get("/metrics")
        
        assert response.status_code == 200
        
        # Parse Prometheus format (metric_name{label="value"} value)
        content = response.text
        lines = content.split("\n")
        
        # Should have at least one metric line (not comment or empty)
        metric_lines = [l for l in lines if l and not l.startswith("#")]
        assert len(metric_lines) > 0
