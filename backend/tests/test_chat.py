from unittest.mock import AsyncMock,patch
from fastapi.testclient import TestClient
from app.main import app
from app.service.exceptions import (
    LLMTimeoutError,
    LLMUnavailableError,
)


client = TestClient(app)

def test_chat_success():
    """Verify that the chat API returns a successful LLM response."""
    with patch(
        "app.api.chat.generate_response",
        new_callable=AsyncMock
    )as mock_llm:
        
        mock_llm.return_value = "Risk-based testing prioritizes testing based on risk."
        
        response = client.post(
            "api/chat",
            json = {
                "message": "What is risk based testing?"
            }
        )
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert "response" in data
        assert "model" in data
        
        assert data["response"] == (
            "Risk-based testing prioritizes testing based on risk."
        )
        
def test_chat_missing_message():
    """Verify that the chat API rejects a request with a missing message."""

    response = client.post(
        "/api/chat",
        json={}
    )

    assert response.status_code == 422
    
def test_chat_invalid_message_type():
    """Verify that the chat API rejects an invalid message data type."""

    response = client.post(
        "/api/chat",
        json={
            "message": 123
        }
    )

    assert response.status_code == 422
    
def test_chat_llm_unavailable():
    """Verify API returns 503 when the LLM service is unavailable."""

    with patch(
        "app.api.chat.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm:

        mock_llm.side_effect = LLMUnavailableError(
            "LLM service is unavailable."
        )

        response = client.post(
            "/api/chat",
            json={
                "message": "Explain test architecture."
            },
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "LLM service unavailable"
        }
        
def test_chat_llm_timeout():
    """Verify API returns 504 when the LLM request times out."""

    with patch(
        "app.api.chat.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm:

        mock_llm.side_effect = LLMTimeoutError(
            "LLM request timed out."
        )

        response = client.post(
            "/api/chat",
            json={
                "message": "Explain test architecture."
            },
        )

        assert response.status_code == 504
        assert response.json() == {
            "detail": "LLM request timed out"
        }