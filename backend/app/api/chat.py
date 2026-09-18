from fastapi import APIRouter

from app.models.chat import ChatRequest, ChatResponse
from app.service.llm_service import generate_response
from app.config.settings import MODEL_NAME

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    response = await generate_response(request.message)

    return ChatResponse(
        response=response,
        model=MODEL_NAME)