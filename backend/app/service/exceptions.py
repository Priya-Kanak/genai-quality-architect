class LLMServiceError(Exception):
    """Base exception for LLM service failures."""
    pass

class LLMTimeoutError(Exception):
    """Raised when the LLM does not respond within the configured timeout."""
    pass

class LLMUnavailableError(Exception):
    """Raised when the LLM service is unavailable."""
    pass