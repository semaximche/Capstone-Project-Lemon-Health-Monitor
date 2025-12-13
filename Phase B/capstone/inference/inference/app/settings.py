from pydantic import Field
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./data_storage/mydb.sqlite3")
    queue_host: str = Field(default="localhost")
    queue_name: str = Field(default="disease_jobs")
    queue_user: str = Field(default="guest")
    queue_password: str = Field(default="guest")

    storage_host: str = Field(default="./data_storage/storage")
    storage_bucket: str = Field(default="analysis", description="Storage bucket name")
    storage_endpoint: str | None = Field(default="users", description="Storage endpoint URL (for S3/Minio)")
    storage_access_key: str | None = Field(default="admin", description="Storage access key")
    storage_secret_key: str | None = Field(default="admin12345", description="Storage secret key")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()