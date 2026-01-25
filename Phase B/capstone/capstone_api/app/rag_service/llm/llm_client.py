from app.settings import settings
from google import genai


def get_llm_client():

    client = genai.Client(
        api_key=settings.gemini_api_key
    )
    return client
