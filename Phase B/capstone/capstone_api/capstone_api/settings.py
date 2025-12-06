"""Application settings and configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
        default="postgresql://postgres:postgres@localhost:5432/capstone",
        description="Database connection URL",
    )

    # Object Storage settings (for image storage)
    storage_type: str = Field(
        default="local",
        description="Storage type: local, s3, minio, or firebase",
    )
    storage_bucket: str = Field(default="capstone-images", description="Storage bucket name")
    storage_endpoint: str | None = Field(default=None, description="Storage endpoint URL (for S3/Minio)")
    storage_access_key: str | None = Field(default=None, description="Storage access key")
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
        default="change-me-in-production",
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
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(default="llama3", description="Ollama model to use")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


settings = get_settings()
