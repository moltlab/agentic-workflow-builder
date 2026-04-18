"""
Media upload API routes for multi-modal agent inputs.
Provides endpoints to upload images/videos to GCS and retrieve metadata.
Stores media metadata in the Media table for association with transactions.
"""

import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from db.config import get_db
from db.crud.media import create_media
from db.crud.session import create_session
from db.models import Session as SessionModel
from utils.media_storage import GCSMediaStorage, MediaValidationError, MediaItem
from utils.logging_utils import get_logger
from utils.user_context import resolve_user_entity

logger = get_logger("media_routes")

media_router = APIRouter(
    prefix="/api/media",
    tags=["media"],
)

_gcs_storage: Optional[GCSMediaStorage] = None


def get_gcs_storage() -> GCSMediaStorage:
    """Lazy-init singleton so the app starts even if GCS is not configured."""
    global _gcs_storage
    if _gcs_storage is None:
        try:
            _gcs_storage = GCSMediaStorage()
        except ValueError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Media storage is not configured: {e}",
            )
    return _gcs_storage


@media_router.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload an image or video file to GCS.

    - Accepts multipart/form-data with a `file` field.
    - Optional `session_id` to group media with a session.
    - Returns media metadata including a signed URL for immediate use.
    """
    storage = get_gcs_storage()

    content_type = file.content_type or storage.detect_content_type(file.filename or "unknown")

    try:
        storage.validate_media_type(content_type)
    except MediaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        file_bytes = await file.read()
        file_size = len(file_bytes)

        storage.validate_file_size(file_size, content_type)
    except MediaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import io

    try:
        media_item = storage.upload(
            file_obj=io.BytesIO(file_bytes),
            filename=file.filename or f"upload_{uuid.uuid4()}",
            content_type=content_type,
            session_id=session_id,
            file_size=file_size,
        )
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    signed_url = storage.get_signed_url(media_item.cloud_path)
    media_item.signed_url = signed_url

    # Ensure session row exists when session_id is provided so FK on media.session_id is valid.
    # This mirrors /ask and /ask-agent-stream behavior: accept client-provided session_id,
    # create the session if it doesn't exist, and generate a new one on invalid format.
    session_uuid = None
    if session_id:
        try:
            session_uuid = uuid.UUID(session_id)
            existing = db.query(SessionModel).filter(SessionModel.id == session_uuid).first()
            if not existing:
                create_session(
                    db,
                    {
                        "id": session_uuid,
                        "start_time": datetime.now(),
                        "status": "active",
                    },
                )
                logger.info(f"Created new session for media upload: {session_uuid}")
        except (ValueError, TypeError):
            # Invalid client session_id – generate a new one and persist it
            session_uuid = uuid.uuid4()
            create_session(
                db,
                {
                    "id": session_uuid,
                    "start_time": datetime.now(),
                    "status": "active",
                },
            )
            session_id = str(session_uuid)
            logger.info(f"Invalid session_id on media upload; generated new session: {session_uuid}")

    user_id, entity_id = resolve_user_entity()

    media_record = create_media(
        db,
        filename=media_item.filename,
        media_type=media_item.media_type,
        cloud_path=media_item.cloud_path,
        file_size=media_item.file_size,
        session_id=session_uuid,
        user_id=user_id,
        entity_id=entity_id,
    )

    logger.info(
        f"Media uploaded: id={media_record.id} type={content_type} "
        f"size={file_size} user=system"
    )

    return {
        "media_id": str(media_record.id),
        "filename": media_item.filename,
        "media_type": media_item.media_type,
        "cloud_path": media_item.cloud_path,
        "file_size": media_item.file_size,
        "signed_url": signed_url,
        "session_id": session_id,
    }


@media_router.get("/{media_id}/url")
async def get_media_url(
    media_id: str,
):
    """
    Generate a fresh signed URL for a previously uploaded media file.
    Requires knowing the cloud_path (returned from upload).

    Note: In a future phase, this can look up the Media DB table by media_id.
    For now, the caller may pass cloud_path as a query param.
    """
    raise HTTPException(
        status_code=501,
        detail="Lookup by media_id will be available after the Media DB table is implemented (Task 5.1). "
        "Use the signed_url returned from /upload for now.",
    )


@media_router.delete("/{media_id}")
async def delete_media(
    media_id: str,
    cloud_path: str,
):
    """Delete a media file from cloud storage by its cloud_path."""
    storage = get_gcs_storage()
    deleted = storage.delete(cloud_path)
    if not deleted:
        raise HTTPException(status_code=404, detail="Media file not found in storage")
    return {"deleted": True, "media_id": media_id}
