
import pytest
from unittest.mock import AsyncMock

from app.rag.rag_service import RAGService


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=3):
        return self.results


@pytest.mark.anyio
async def test_rag_returns_grounded_answer(monkeypatch):
    chunks = [
        {
            "text": "Production access is reviewed every 90 days.",
            "metadata": {
                "source": "policy_document.pdf",
                "page": 3,
                "chunk_id": "policy_document.pdf_p3_c1",
            },
            "distance": 0.12,
        }
    ]

    mock_generate = AsyncMock(
        return_value="Every 90 days."
    )

    monkeypatch.setattr(
        "app.rag.rag_service.generate_response",
        mock_generate,
    )

    service = RAGService(FakeRetriever(chunks))

    result = await service.ask(
        "How often is production access reviewed?"
    )

    assert result["answer"] == "Every 90 days."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 3

    mock_generate.assert_awaited_once()


@pytest.mark.anyio
async def test_rag_rejects_unsupported_question(monkeypatch):
    chunks = [
        {
            "text": "Audit logs are retained for 365 days.",
            "metadata": {
                "source": "policy_document.pdf",
                "page": 2,
                "chunk_id": "policy_document.pdf_p2_c1",
            },
            "distance": 0.85,
        }
    ]

    mock_generate = AsyncMock()

    monkeypatch.setattr(
        "app.rag.rag_service.generate_response",
        mock_generate,
    )

    service = RAGService(FakeRetriever(chunks))

    result = await service.ask(
        "What is the CEO's personal phone number?"
    )

    assert "couldn't find sufficient information" in result["answer"]
    assert result["sources"] == []

    mock_generate.assert_not_awaited()
    
    
@pytest.fixture
def anyio_backend():
    return "asyncio"

