"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Find .env file - check project root first, then current directory
# This works for both local dev (from project root) and Docker (from container working dir)
_env_file_paths = [
    Path(__file__).parent.parent.parent / ".env",  # Project root: capstone_api/app/settings.py -> root/.env
    Path(".env"),  # Current directory (for Docker or if running from root)
]

_env_file = None
for path in _env_file_paths:
    if path.exists():
        _env_file = str(path)
        break


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = Field(default="Capstone API", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Database settings
    database_url: str = Field(
        default="sqlite:///../data_storage/mydb.sqlite3",
        description="Database connection URL",
    )

    # Object Storage settings (for image storage)
    storage_host: str = Field(
        default="../data_storage/storage",
        description="Storage type: local, s3, minio, or firebase",
    )
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    storage_bucket: str = Field(default="analysis", description="Storage bucket name")
    storage_endpoint: str | None = Field(default="users", description="Storage endpoint URL (for S3/Minio)")
    storage_access_key: str | None = Field(default="admin", description="Storage access key")
    storage_secret_key: str | None = Field(default=None, description="Storage secret key")

    # Google OAuth settings
    google_client_id: str | None = Field(default=None, description="Google OAuth client ID")
    google_client_secret: str | None = Field(default=None, description="Google OAuth client secret")
    google_redirect_uri: str = Field(
        default="http://localhost:8000/v1/auth/callback",
        description="Google OAuth redirect URI",
    )

    # JWT settings
    jwt_secret_key: str = Field(
        default="",
        description="Secret key for JWT token signing",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration time in days",
    )

    # Home Assistant settings
    home_assistant_url: str | None = Field(
        default=None,
        description="Home Assistant instance URL",
    )
    home_assistant_token: str | None = Field(
        default=None,
        description="Home Assistant long-lived access token",
    )

    # ML Pipeline settings
    model_device: str = Field(
        default="cpu",
        description="Device for running ML models (cpu, cuda, mps)",
    )
    yolo_model_path: str | None = Field(
        default=None,
        description="Path to YOLOv11 model weights",
    )
    classification_model_path: str | None = Field(
        default=None,
        description="Path to disease classification model weights",
    )

    # LLM settings
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai or ollama",
    )
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(default="llama3", description="Ollama model to use")

    # RAG settings
    rag_vector_db_path: str = Field(
        default="../data_storage/vector_db",
        description="Path to vector database storage",
    )
    rag_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    rag_chunk_size: int = Field(default=800, description="Text chunk size for document processing")
    rag_chunk_overlap: int = Field(default=150, description="Overlap between text chunks")
    rag_collection_name: str = Field(
        default="project_book",
        description="ChromaDB collection name",
    )
    rag_top_k: int = Field(default=5, description="Number of top results to retrieve from vector search")

    # Rabbitmq settings
    queue_host: str = Field(default="localhost", description="Rabbitmq server host")
    analysis_queue_name: str = Field(default="disease_jobs", description="Rabbitmq queue name")
    notifications_queue_name: str = Field(default="notifications", description="notifications Rabbitmq queue name")
    events_exchange: str = Field(default="events_exchange", description="notifications Rabbitmq queue name")
    queue_user: str = Field(default="guest", description="RabbitMQ username")
    queue_password: str = Field(default="guest", description="RabbitMQ password")

settings = Settings()
