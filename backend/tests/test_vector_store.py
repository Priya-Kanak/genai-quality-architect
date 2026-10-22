from app.rag.vector_store import VectorStore


def test_add_and_search_chunks(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )

    chunks = [
        {
            "text": "Production access is reviewed every 90 days.",
            "metadata": {
                "source": "test.pdf",
                "page": 1,
                "chunk_id": "test_p1_c1",
                "chunk_number": 1,
            },
        },
        {
            "text": "Lost devices must be reported within one hour.",
            "metadata": {
                "source": "test.pdf",
                "page": 2,
                "chunk_id": "test_p2_c1",
                "chunk_number": 1,
            },
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results["documents"][0][0] == (
        "Production access is reviewed every 90 days."
    )

    assert (
        results["metadatas"][0][0]["source"] == "test.pdf"
    )
