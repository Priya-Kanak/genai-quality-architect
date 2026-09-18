"""Tests for the application health-check endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health():
    """Verify that the health endpoint returns UP status."""

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"
   