from sqlalchemy.orm import Session
from db.models import AgentExecutionLog
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api.schemas.auth import CurrentUser


def create_agent_execution_log(
    db: Session,
    data: dict,
    current_user: Optional["CurrentUser"] = None,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None,
):
    """
    Create a new agent execution log with user/entity context.

    Args:
        db: Database session
        data: Dictionary with agent execution log data
        current_user: Optional CurrentUser for auth context
        user_id: Optional direct user_id (takes precedence)
        entity_id: Optional direct entity_id (takes precedence)
    """
    from utils.user_context import (
        enrich_data_with_user_context,
        get_user_context_from_transaction,
        resolve_user_entity,
    )

    # Priority 1: If user_id/entity_id provided directly, use them
    if user_id and entity_id:
        data["user_id"] = user_id
        data["entity_id"] = entity_id
    # Priority 2: Try to get from transaction if available
    elif data.get("transaction_id"):
        try:
            user_id_from_tx, entity_id_from_tx = get_user_context_from_transaction(
                db, data["transaction_id"]
            )
            data["user_id"] = user_id_from_tx
            data["entity_id"] = entity_id_from_tx
        except ValueError:
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
        user_id_default, entity_id_default = resolve_user_entity(None)
        data.setdefault("user_id", user_id_default)
        data.setdefault("entity_id", entity_id_default)

    record = AgentExecutionLog(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_agent_execution_log_by_id(db: Session, id):
    return db.query(AgentExecutionLog).filter(AgentExecutionLog.id == id).first()

def list_agent_execution_logs(db: Session):
    return db.query(AgentExecutionLog).all()