
from pathlib import Path

from app.rag.document_loader import load_pdf
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore


def main():
    backend_dir = Path(__file__).resolve().parent

    pdf_path = (
        backend_dir
        / "data"
        / "documents"
        / "policy_document.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Policy document not found: {pdf_path}"
        )

    print("Step 1: Loading policy PDF...")
    pages = load_pdf(str(pdf_path))
    print(f"Pages loaded: {len(pages)}")

    print("Step 2: Chunking documents...")
    chunks = chunk_documents(
        pages,
        chunk_size=120,
        chunk_overlap=20,
    )
    print(f"Chunks created: {len(chunks)}")

    print("Step 3: Generating embeddings...")
    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    print(f"Embeddings generated: {len(embeddings)}")

    print("Step 4: Storing chunks in ChromaDB...")
    store = VectorStore(
        persist_directory=str(backend_dir / "data" / "chroma"),
        collection_name="policy_documents",
    )

    store.add_chunks(chunks, embeddings)

    print(f"Indexed chunks: {store.collection.count()}")
    print("Document ingestion completed successfully.")


if __name__ == "__main__":
    main()
