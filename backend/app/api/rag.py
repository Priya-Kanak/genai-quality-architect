from fastapi import APIRouter,HTTPException,Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["RAG"])

rag_service = None

class RAGRequest(BaseModel):
    question: str = Field(min_length=1)

@router.post("/rag/ask")
async def ask_rag(payload: RAGRequest, request: Request):
    rag_service = getattr(
        request.app.state,
        "rag_service",
        None,
    )
    
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized"
        )

    return await rag_service.ask(payload.question)
