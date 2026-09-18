import httpx

from app.config.settings import OLLAMA_URL, MODEL_NAME

async def generate_response(message: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": message,
        "stream": False
    }

    async with httpx.AsyncClient(timeout = 120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json = payload
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]
