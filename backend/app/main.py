
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.rag import router as rag_router

from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend_dir = Path(__file__).resolve().parents[1]

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        persist_directory=str(
            backend_dir / "data" / "chroma"
        ),
        collection_name="policy_documents",
    )

    retriever = Retriever(
        embedding_model=embedding_service,
        vector_store=vector_store,
    )

    app.state.rag_service = RAGService(retriever)

    print(
        "RAG initialized. Indexed chunks:",
        vector_store.collection.count(),
    )

    yield

    app.state.rag_service = None


app = FastAPI(
    title="GenAI Quality Architecture",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "UP"}


app.include_router(chat_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
