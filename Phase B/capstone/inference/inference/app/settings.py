from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict
import os
from pathlib import Path

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

    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str = Field(default="sqlite:///../data_storage/mydb.sqlite3")
    queue_host: str = Field(default="rabbitmq")
    analysis_queue_name: str = Field(default="disease_jobs")
    notifications_queue_name: str = Field(default="notifications")
    events_exchange: str = Field(default="events_exchange", description="notifications Rabbitmq queue name")
    queue_user: str = Field(default="guest")
    queue_password: str = Field(default="guest")
    gemini_api_key:str = Field(default="")
    storage_host: str = Field(default="../data_storage/storage")
    storage_bucket: str = Field(default="analysis", description="Storage bucket name")
    storage_endpoint: str | None = Field(default="users", description="Storage endpoint URL (for S3/Minio)")
    storage_access_key: str | None = Field(default="admin", description="Storage access key")
    storage_secret_key: str | None = Field(default="admin12345", description="Storage secret key")


settings = Settings()