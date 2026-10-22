from app.rag.retriever import Retriever
import pytest


class FakeEmbeddingModel:
    def encode(self, query):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def search(self, query_embedding, top_k):
        return {
            "documents": [["Production access is reviewed every 90 days."]],
            "metadatas": [
                [
                    {
                        "source": "policy.pdf",
                        "page": 2,
                        "chunk_id": "policy_p2_c1",
                    }
                ]
            ],
            "distances": [[0.05]],
        }


def test_retriever_returns_relevant_chunks():
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="How often is production access reviewed?",
        top_k=1,
    )

    assert len(results) == 1

    assert results[0]["text"] == ("Production access is reviewed every 90 days.")

    assert results[0]["metadata"]["source"] == "policy.pdf"
    assert results[0]["metadata"]["page"] == 2
    assert results[0]["distance"] == 0.05
      
def test_retriever_rejects_empty_query():
    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve(query="", top_k=1)


def test_retriever_rejects_invalid_top_k():
    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="top_k must be atleast 1"):
        retriever.retrieve(
            query="How often is production access reviewed?",
            top_k=0,
        )
