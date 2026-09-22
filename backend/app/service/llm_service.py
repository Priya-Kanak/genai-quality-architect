import asyncio

import httpx

from app.config.logging_config import logger
from app.config.settings import MODEL_NAME, OLLAMA_URL
from app.service.exceptions import (
    LLMTimeoutError,
    LLMUnavailableError,
)


MAX_RETRIES = 3


async def generate_response(message: str) -> str:
    """Generate a response from the configured LLM with retry handling."""

    payload = {
        "model": MODEL_NAME,
        "prompt": message,
        "stream": False,
    }

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "Calling LLM model=%s attempt=%s",
                MODEL_NAME,
                attempt + 1,
            )

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                return data["response"]

        except httpx.TimeoutException as exc:
            logger.warning(
                "LLM timeout attempt=%s",
                attempt + 1,
            )

            if attempt == MAX_RETRIES - 1:
                raise LLMTimeoutError(
                    "LLM request timed out."
                ) from exc

        except (
            httpx.ConnectError,
            httpx.NetworkError,
        ) as exc:
            logger.error(
                "LLM unavailable attempt=%s",
                attempt + 1,
            )

            if attempt == MAX_RETRIES - 1:
                raise LLMUnavailableError(
                    "LLM service is unavailable."
                ) from exc

        await asyncio.sleep(2 ** attempt)

    raise LLMUnavailableError(
        "LLM service failed after maximum retries."
    )