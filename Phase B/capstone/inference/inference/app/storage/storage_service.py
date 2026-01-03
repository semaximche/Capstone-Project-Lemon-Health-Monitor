import os
import shutil
from pathlib import Path
from typing import List
from inference.app.settings import settings

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

    def upload_file(
            self,
            object_name: str,
            source_path: str | Path,
            extension: str | None = None,
    ) -> Path:
        """
        Save a file into local storage and return its final path.
        This method ensures that the object_name is correctly saved within the desired folder structure.
        """

        # Convert source_path to a Path object for better handling
        source_path = Path(source_path)

        # Normalize object_name to use forward slashes (Unix-style paths)
        # but keep it compatible for Windows paths
        object_name = object_name.replace("\\", "/")

        # Ensure the object_name doesn't contain dangerous or invalid characters
        object_path = Path(object_name)
        if object_path.is_absolute() or ".." in object_path.parts:
            raise ValueError("Invalid object_name path: path traversal is not allowed")

        # Ensure that the extension is properly handled
        if extension:
            # Enforce the extension starts with a dot
            if not extension.startswith("."):
                extension = f".{extension}"
            # Ensure the extension is added if it's not already part of the object_name
            if not object_name.endswith(extension):
                object_name += extension

        # Build the final destination path using the base path
        dest_path = self.full_path(object_path)

        # Create the parent directories if they don't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the source file exists before copying it
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Copy the file to the destination path
        shutil.copy2(source_path, dest_path)

        # Return the destination path for further usage
        return dest_path


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