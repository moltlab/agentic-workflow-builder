from sqlalchemy.orm import Session
from db.models import LLMUsageLog
from typing import Optional
from api.schemas.auth import CurrentUser

def create_llm_usage_log(
    db: Session, 
    data: dict,
    current_user: Optional[CurrentUser] = None,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None
):
    """
    Create a new LLM usage log with user/entity context.
    
    Args:
        db: Database session
        data: Dictionary with LLM usage log data
        current_user: Optional CurrentUser for auth context
        user_id: Optional direct user_id (takes precedence)
        entity_id: Optional direct entity_id (takes precedence)
    """
    from utils.user_context import enrich_data_with_user_context, get_user_context_from_transaction, resolve_user_entity
    
    # Priority 1: If user_id/entity_id provided directly, use them
    if user_id and entity_id:
        data["user_id"] = user_id
        data["entity_id"] = entity_id
    # Priority 2: Try to get from transaction if available
    elif data.get("transaction_id"):
        try:
            user_id_from_tx, entity_id_from_tx = get_user_context_from_transaction(db, data["transaction_id"])
            data["user_id"] = user_id_from_tx
            data["entity_id"] = entity_id_from_tx
        except ValueError:
            # Transaction not found, use current_user or defaults
            if current_user:
                data = enrich_data_with_user_context(data, current_user=current_user)
            else:
                user_id_default, entity_id_default = resolve_user_entity(None)
                data.setdefault("user_id", user_id_default)
                data.setdefault("entity_id", entity_id_default)
    # Priority 3: Use current_user if available
    elif current_user:
        data = enrich_data_with_user_context(data, current_user=current_user)
    else:
        # Last resort: use defaults
        user_id_default, entity_id_default = resolve_user_entity(None)
        data.setdefault("user_id", user_id_default)
        data.setdefault("entity_id", entity_id_default)
    
    record = LLMUsageLog(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_llm_usage_log_by_id(db: Session, id):
    return db.query(LLMUsageLog).filter(LLMUsageLog.id == id).first()

def list_llm_usage_logs(db: Session):
    return db.query(LLMUsageLog).all()