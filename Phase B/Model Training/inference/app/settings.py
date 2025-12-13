from pydantic import Field

class Settings:

    database_url: str = "sqlite:///C:/projects/mydb.sqlite3" #for regular run
    queue_host: str = "localhost" #for regular run
    # queue_host: str = "rabbitmq" #for docker

    storage_host: str = "C:\projects\data"
    storage_bucket: str = Field(default="analysis", description="Storage bucket name")
    storage_endpoint: str | None = Field(default="users", description="Storage endpoint URL (for S3/Minio)")
    storage_access_key: str | None = Field(default="admin", description="Storage access key")
    storage_secret_key: str | None = Field(default="admin12345", description="Storage secret key")


settings = Settings()