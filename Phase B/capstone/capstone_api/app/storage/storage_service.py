import os
import shutil
from pathlib import Path
from typing import List
from app.settings import settings

class FileSystemStorageService:
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


    def full_path(self, object_name: str) -> Path:
        """
        Convert object_name -> full filesystem path.
        Example:
        object_name="users/123/uploads/file.jpg"
        => storage/users/123/uploads/file.jpg
        """
        return self.base_path / object_name

    def ensure_dir(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def upload_file(self, object_name: str, file_path: str) -> Path:
        """
        Copy file from local filesystem into storage.
        """
        dest = self.full_path(object_name)
        self.ensure_dir(dest)

        shutil.copy(file_path, dest)
        return dest



    def download_file(self, object_name: str, download_path: str):
        """
        Copy stored file to another local path.
        """
        src = self.full_path(object_name)
        shutil.copy(src, download_path)

    def delete_object(self, object_name: str):
        """
        Delete a file.
        """
        path = self.full_path(object_name)
        if path.exists():
            path.unlink()

    def list_objects(self, prefix: str = "") -> List[str]:
        """
        List all objects (recursive) under a prefix.
        Example: prefix="users/123/uploads/"
        """
        root = self.base_path / prefix
        if not root.exists():
            return []

        result = []
        for file in root.rglob("*"):
            if file.is_file():
                result.append(str(file.relative_to(self.base_path)))

        return result

    def object_exists(self, object_name: str) -> bool:
        """
        Check if file exists.
        """
        return self.full_path(object_name).exists()


storage_service = FileSystemStorageService(settings.storage_host)