from minio import Minio
from threading import Lock
from inference.app.settings import settings

class MinioClient:
    _instance = None
    _lock = Lock()
    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MinioClient, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 endpoint: str = settings.storage_host,
                 access_key: str = settings.storage_access_key,
                 secret_key: str = settings.storage_secret_key,
                 secure: bool = False):
        # Only initialize once
        if not hasattr(self, "client"):
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )

    def get_client(self) -> Minio:
        return self.client
