import logging

import pytest
from fastapi.testclient import TestClient

from src.api import app, predictor


class FakePipeline:
    def predict(self, texts):
        return [1]

    def predict_proba(self, texts):
        return [[0.08, 0.92]]


class BrokenPipeline(FakePipeline):
    def predict(self, texts):
        raise ValueError("Internal model detail")


@pytest.fixture
def client(monkeypatch):
    previous_pipeline = predictor.pipeline

    def fake_load():
        predictor.pipeline = FakePipeline()

    monkeypatch.setattr(predictor, "load", fake_load)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        predictor.pipeline = previous_pipeline


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_health_reports_unavailable_model(client):
    predictor.pipeline = None

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "model_loaded": False}


def test_predict_reports_unavailable_model(client):
    predictor.pipeline = None

    response = client.post("/predict", json={"text": "Great movie"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction service is temporarily unavailable"
    }


def test_valid_review_returns_prediction(client):
    response = client.post("/predict", json={"text": "  Great movie  "})

    assert response.status_code == 200
    assert response.json() == {
        "text": "Great movie",
        "sentiment": "positive",
        "confidence": 0.92,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 123},
        {"text": "x" * 10_001},
    ],
)
def test_invalid_review_returns_422(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_model_failure_returns_safe_503(client):
    predictor.pipeline = BrokenPipeline()

    response = client.post("/predict", json={"text": "Great movie"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction service is temporarily unavailable"
    }
    assert "Internal model detail" not in response.text


def test_docs_are_available(client):
    response = client.get("/docs")

    assert response.status_code == 200


def test_request_logs_status_and_duration_without_review_text(client, caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        success = client.post("/predict", json={"text": "Private review text"})
        invalid = client.post("/predict", json={"text": "   "})

    assert success.status_code == 200
    assert invalid.status_code == 422
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "uvicorn.error"
    ]
    assert any("method=POST path=/predict status=200 duration_ms=" in message for message in messages)
    assert any("method=POST path=/predict status=422 duration_ms=" in message for message in messages)
    assert all("Private review text" not in message for message in messages)
