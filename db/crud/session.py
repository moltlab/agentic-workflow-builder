from sqlalchemy.orm import Session
from db.models import Session as SessionModel
from typing import Optional
from api.schemas.auth import CurrentUser

def create_session(db: Session, data: dict, current_user: Optional[CurrentUser] = None):
    """
    Create a new session with user/entity context.
    
    Args:
        db: Database session
        data: Dictionary with session data
        current_user: Optional CurrentUser for auth context
        
    Security:
        Never trusts client-provided entity_id. Always derives from auth context.
    """
    from utils.user_context import enrich_data_with_user_context
    
    # Enrich with user/entity context (removes client-provided values)
    data = enrich_data_with_user_context(data, current_user=current_user)
    
    record = SessionModel(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_session_by_id(db: Session, id):
    return db.query(SessionModel).filter(SessionModel.id == id).first()

def list_sessions(db: Session):
    return db.query(SessionModel).all()

def session_exists(db: Session, id):
    return db.query(SessionModel).filter(SessionModel.id == id).first() is not None