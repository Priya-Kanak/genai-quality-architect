from pathlib import Path

from app.rag.document_loader import load_pdf
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever

def test_pdf_retrieval_pipeline(tmp_path):
    #1. locate the real policy pdf
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "documents"
        / "policy_document.pdf"
    )
    
    assert pdf_path.exists(), "Policy PDF not found"
    
    #2. Load PDF pages
    pages = load_pdf(str(pdf_path))
    
    assert len(pages) > 0
    
    #3. Split pages into chunks
    chunks = chunk_documents(
        pages,
        chunk_size=120,
        chunk_overlap= 20,
    )
    
    assert len(chunks)>0
    
    # 4. Generate real document embeddings
    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    assert len(embeddings) == len(chunks)

    # 5. Store chunks in isolated ChromaDB
    store = VectorStore(
        persist_directory=str(tmp_path),
        collection_name="pdf_retrieval_test",
    )

    store.add_chunks(chunks, embeddings)

    # 6. Create Retriever
    retriever = Retriever(
        embedding_model=embedding_service,
        vector_store=store,
    )

    # 7. Retrieve evidence
    results = retriever.retrieve(
        query="How frequently is production access reviewed?",
        top_k=3,
    )

    # 8. Validate retrieval output
    assert len(results) > 0
    assert len(results) <= 3
    
    # Verify expected evidence is ranked first
    assert (
    results[0]["metadata"]["chunk_id"]
    == "policy_document.pdf_p3_c1"
    )

    assert (
    "Production access must be reviewed every 90 days."
    in results[0]["text"]
    )

    for result in results:
        assert result["text"].strip()
        assert result["metadata"]["source"] == "policy_document.pdf"
        assert result["metadata"]["page"] >= 1
        assert result["metadata"]["chunk_id"]

    # 9. Print retrieved evidence for inspection
    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Text:", result["text"])
        print("Metadata:", result["metadata"])
        print("Cosine distance:", result["distance"])