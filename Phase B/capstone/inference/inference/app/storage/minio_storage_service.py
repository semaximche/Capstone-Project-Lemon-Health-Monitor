# from datetime import timedelta
# from io import BytesIO
# from minio.error import S3Error
#
# from inference.app.storage.minio_storage_client import MinioClient
#
#
# class MinioStorageService:
#     def __init__(self):
#         self.client = MinioClient().get_client()
#
#
#     def ensure_bucket(self, bucket_name: str):
#         """Create bucket if not exists."""
#         if not self.client.bucket_exists(bucket_name):
#             self.client.make_bucket(bucket_name)
#
#
#     def upload_file(self, bucket: str, object_name: str, file_path: str, content_type: str = "application/octet-stream"):
#         """Upload file from local path to MinIO."""
#         self.ensure_bucket(bucket)
#         self.client.fput_object(
#             bucket_name=bucket,
#             object_name=object_name,
#             file_path=file_path,
#             content_type=content_type
#         )
#
#     def download_file(self, bucket: str, object_name: str, download_path: str):
#         """Download from MinIO to local file."""
#         self.client.fget_object(bucket, object_name, download_path)
#
#     def download_bytes(self, bucket: str, object_name: str) -> bytes:
#         """Download object as bytes."""
#         response = self.client.get_object(bucket, object_name)
#         data = response.read()
#         response.close()
#         response.release_conn()
#         return data
#
#     def delete_object(self, bucket: str, object_name: str):
#         """Delete a single object."""
#         self.client.remove_object(bucket, object_name)
#
#     def list_objects(self, bucket: str, prefix: str = ""):
#         """List objects under specific prefix."""
#         return [obj.object_name for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True)]
#
#     def object_exists(self, bucket: str, object_name: str) -> bool:
#         """Check if object exists."""
#         try:
#             self.client.stat_object(bucket, object_name)
#             return True
#         except S3Error as e:
#             if e.code == "NoSuchKey":
#                 return False
#             raise e
#
#
#     def get_presigned_url(self, bucket: str, object_name: str) -> str:
#         """get presigned url for object"""
#         return self.client.presigned_get_object(
#             bucket_name=bucket,
#             object_name=object_name,
#         )
#
#
#
# minio_storage_service = MinioStorageService()