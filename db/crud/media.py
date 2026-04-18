"""CRUD operations for Media table (multi-modal uploads)."""

from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from db.models import Media


def create_media(
    db: Session,
    *,
    filename: str,
    media_type: str,
    cloud_path: str,
    file_size: int,
    session_id: Optional[UUID] = None,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> Media:
    """Create a Media record. Returns the created row with id."""
    record = Media(
        filename=filename,
        media_type=media_type,
        cloud_path=cloud_path,
        file_size=file_size,
        session_id=session_id,
        user_id=user_id,
        entity_id=entity_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_media_by_id(db: Session, media_id: UUID) -> Optional[Media]:
    """Get a single Media by id."""
    return db.query(Media).filter(Media.id == media_id).first()


def get_media_by_ids(db: Session, media_ids: List[UUID], session_id: Optional[UUID] = None) -> List[Media]:
    """
    Get Media rows by ids. Optionally filter by session_id for scoping.
    Returns in the same order as media_ids where found.
    """
    if not media_ids:
        return []
    query = db.query(Media).filter(Media.id.in_(media_ids))
    if session_id is not None:
        query = query.filter(Media.session_id == session_id)
    rows = query.all()
    order_map = {r.id: r for r in rows}
    return [order_map[mid] for mid in media_ids if mid in order_map]


def update_media_transaction_id(db: Session, media_ids: List[UUID], transaction_id: UUID) -> int:
    """Link Media rows to a transaction. Returns count updated."""
    if not media_ids:
        return 0
    updated = db.query(Media).filter(Media.id.in_(media_ids)).update(
        {"transaction_id": transaction_id},
        synchronize_session="fetch",
    )
    db.commit()
    return updated
