"""
Google Cloud Storage media storage service for multi-modal agent inputs.
Handles upload, signed URL generation, validation, and cleanup of media files.
"""

import os
import uuid
import mimetypes
from datetime import timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, BinaryIO

from google.cloud import storage
from google.oauth2 import service_account

from utils.logging_utils import get_logger

logger = get_logger("media_storage")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
ALLOWED_MEDIA_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


@dataclass
class MediaItem:
    """Represents a media file in cloud storage (path is cloud-agnostic: GCS, S3, Azure, etc.)."""
    media_id: str
    filename: str
    media_type: str
    cloud_path: str  # path/key in cloud storage
    file_size: int
    session_id: Optional[str] = None
    signed_url: Optional[str] = field(default=None, repr=False)

    @property
    def is_image(self) -> bool:
        return self.media_type in ALLOWED_IMAGE_TYPES

    @property
    def is_video(self) -> bool:
        return self.media_type in ALLOWED_VIDEO_TYPES

    def to_dict(self) -> dict:
        return asdict(self)


class MediaValidationError(Exception):
    """Raised when media validation fails."""
    pass


_storage_instance: Optional["GCSMediaStorage"] = None


def get_media_storage() -> Optional["GCSMediaStorage"]:
    """Return storage instance if configured (e.g. GCS); None otherwise. Safe for use from agents/routes."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance
    try:
        _storage_instance = GCSMediaStorage()
        return _storage_instance
    except ValueError:
        return None


class GCSMediaStorage:
    """
    Manages media file lifecycle in Google Cloud Storage.

    Usage:
        storage = GCSMediaStorage()
        item = storage.upload(file_obj, "photo.png", "image/png", session_id="abc")
        url  = storage.get_signed_url(item.cloud_path)
        storage.delete(item.cloud_path)
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        signed_url_expiry_minutes: Optional[int] = None,
        media_prefix: Optional[str] = None,
    ):
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "")
        self.credentials_path = credentials_path or os.getenv("GCS_CREDENTIALS_PATH", "")
        self.signed_url_expiry_minutes = signed_url_expiry_minutes or int(
            os.getenv("GCS_SIGNED_URL_EXPIRY_MINUTES", "60")
        )
        self.media_prefix = media_prefix or os.getenv("GCS_MEDIA_PREFIX", "media")

        if not self.bucket_name:
            raise ValueError(
                "GCS_BUCKET_NAME is required. Set it in .env or pass bucket_name to constructor."
            )

        self._client = self._build_client()
        self._bucket = self._client.bucket(self.bucket_name)
        logger.info(f"GCSMediaStorage initialized — bucket={self.bucket_name}")

    def _build_client(self) -> storage.Client:
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            return storage.Client(credentials=credentials)
        # Fall back to Application Default Credentials
        return storage.Client()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_media_type(content_type: str) -> None:
        if content_type not in ALLOWED_MEDIA_TYPES:
            raise MediaValidationError(
                f"Unsupported media type: {content_type}. "
                f"Allowed: {sorted(ALLOWED_MEDIA_TYPES)}"
            )

    @staticmethod
    def validate_file_size(size_bytes: int, content_type: str) -> None:
        if content_type in ALLOWED_VIDEO_TYPES:
            limit = MAX_VIDEO_SIZE_BYTES
            label = "100 MB"
        else:
            limit = MAX_IMAGE_SIZE_BYTES
            label = "20 MB"

        if size_bytes > limit:
            raise MediaValidationError(
                f"File size {size_bytes / (1024*1024):.1f} MB exceeds "
                f"the {label} limit for {content_type}."
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _build_gcs_path(self, filename: str, session_id: Optional[str] = None) -> tuple[str, str]:
        """Returns (media_id, gcs_object_path)."""
        media_id = str(uuid.uuid4())
        session_part = session_id or "no-session"
        safe_filename = filename.replace(" ", "_")
        gcs_path = f"{self.media_prefix}/{session_part}/{media_id}/{safe_filename}"
        return media_id, gcs_path

    def upload(
        self,
        file_obj: BinaryIO,
        filename: str,
        content_type: str,
        session_id: Optional[str] = None,
        file_size: Optional[int] = None,
    ) -> MediaItem:
        """
        Validate and upload a file to GCS.

        Args:
            file_obj: Readable binary stream (e.g. FastAPI UploadFile.file).
            filename: Original filename.
            content_type: MIME type (e.g. "image/png").
            session_id: Optional session id for path grouping.
            file_size: Known size in bytes (if available). When None, the
                       file is read into memory to measure size.

        Returns:
            MediaItem with all metadata populated.

        Raises:
            MediaValidationError: If type or size validation fails.
        """
        self.validate_media_type(content_type)

        if file_size is not None:
            self.validate_file_size(file_size, content_type)

        media_id, gcs_path = self._build_gcs_path(filename, session_id)
        blob = self._bucket.blob(gcs_path)

        if file_size is not None:
            blob.upload_from_file(file_obj, content_type=content_type)
            actual_size = file_size
        else:
            data = file_obj.read()
            actual_size = len(data)
            self.validate_file_size(actual_size, content_type)
            blob.upload_from_string(data, content_type=content_type)

        logger.info(
            f"Uploaded {filename} ({actual_size} bytes) → gs://{self.bucket_name}/{gcs_path}"
        )

        return MediaItem(
            media_id=media_id,
            filename=filename,
            media_type=content_type,
            cloud_path=gcs_path,
            file_size=actual_size,
            session_id=session_id,
        )

    def get_signed_url(self, cloud_path: str, expiry_minutes: Optional[int] = None) -> str:
        """Generate a time-limited signed URL for the object at cloud_path."""
        minutes = expiry_minutes or self.signed_url_expiry_minutes
        blob = self._bucket.blob(cloud_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=minutes),
            method="GET",
        )
        logger.debug(f"Signed URL generated for {cloud_path} (expires in {minutes}m)")
        return url

    def delete(self, cloud_path: str) -> bool:
        """Delete a single object at cloud_path. Returns True if deleted."""
        blob = self._bucket.blob(cloud_path)
        if blob.exists():
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{cloud_path}")
            return True
        logger.warning(f"Object not found for deletion: {cloud_path}")
        return False

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix (e.g. a session). Returns count deleted."""
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        if not blobs:
            return 0
        self._bucket.delete_blobs(blobs)
        logger.info(f"Deleted {len(blobs)} objects under prefix {prefix}")
        return len(blobs)

    def exists(self, cloud_path: str) -> bool:
        return self._bucket.blob(cloud_path).exists()

    def get_public_url(self, cloud_path: str) -> str:
        """
        Returns the public URL (only works if the object/bucket is publicly readable).
        Prefer get_signed_url() for private buckets.
        """
        return f"https://storage.googleapis.com/{self.bucket_name}/{cloud_path}"

    def detect_content_type(self, filename: str) -> str:
        """Guess MIME type from filename extension."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"
