from fastapi import APIRouter,HTTPException

from app.models.chat import ChatRequest, ChatResponse
from app.service.llm_service import generate_response
from app.config.settings import MODEL_NAME
from app.service.exceptions import(
    LLMTimeoutError,
    LLMUnavailableError,
)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        

        response = await generate_response(request.message)

        return ChatResponse(
             response=response,
             model=MODEL_NAME
        )
    except LLMTimeoutError as exc:    
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out"
        )from exc
        
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable"
        )from exc
         
    
    
    