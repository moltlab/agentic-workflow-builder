from sqlalchemy.orm import Session
from db.models import Tool
from typing import Optional
from api.schemas.auth import CurrentUser

def create_tool(db: Session, data: dict, current_user: Optional[CurrentUser] = None):
    """
    Create a new tool with user/entity context.
    
    Args:
        db: Database session
        data: Dictionary with tool data
        current_user: Optional CurrentUser for auth context
        
    Security:
        Never trusts client-provided entity_id. Always derives from auth context.
    """
    from utils.user_context import enrich_data_with_user_context
    
    # Enrich with user/entity context (removes client-provided values)
    data = enrich_data_with_user_context(
        data, 
        current_user=current_user,
        user_id_field="created_by_user_id",
        include_user_id=True,
        include_entity_id=True
    )
    
    record = Tool(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_tool_by_id(db: Session, id):
    return db.query(Tool).filter(Tool.id == id).first()

def list_tools(db: Session):
    return db.query(Tool).all()