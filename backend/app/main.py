from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(
    title = "GenAI Quality Architecture",
    version = "1.0.0"
)

@app.get("/health")
async def health():

    return{
        "status": "UP"
    }

app.include_router(
    chat_router,
    prefix="/api"
)