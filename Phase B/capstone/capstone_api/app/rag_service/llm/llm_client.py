from app.settings import settings
from google import genai


def get_llm_client():
    """Get Google Gemini LLM client with API key validation."""
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please set it in your .env file."
        )
    
    client = genai.Client(
        api_key=settings.gemini_api_key
    )
    return client
