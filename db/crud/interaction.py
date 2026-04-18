from sqlalchemy.orm import Session
from db.models import Interaction
from collections import defaultdict
from sqlalchemy import or_
from typing import Optional

# Create a new interaction (message)
def create_interaction(
    db: Session, 
    data: dict, 
    current_user=None,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None
):
    """
    Create a new interaction with user/entity context.
    
    Args:
        db: Database session
        data: Dictionary with interaction data
        current_user: Optional CurrentUser for auth context
        user_id: Optional direct user_id (takes precedence over current_user)
        entity_id: Optional direct entity_id (takes precedence over current_user)
        
    Security:
        Never trusts client-provided entity_id. Always derives from auth context.
    """
    from utils.user_context import enrich_data_with_user_context, resolve_user_entity
    from db.models import Session as SessionModel
    
    # Priority 1: If user_id/entity_id provided directly, use them
    if user_id and entity_id:
        data["user_id"] = user_id
        data["entity_id"] = entity_id
    # Priority 2: Use current_user if available
    elif current_user:
        data = enrich_data_with_user_context(data, current_user=current_user)
    # Priority 3: Try to derive from session
    elif data.get("session_id"):
        session = db.query(SessionModel).filter(
            SessionModel.id == data["session_id"]
        ).first()
        if session and session.user_id and session.entity_id:
            data["user_id"] = session.user_id
            data["entity_id"] = session.entity_id
        else:
            # Last resort: use defaults
            user_id_default, entity_id_default = resolve_user_entity(None)
            data.setdefault("user_id", user_id_default)
            data.setdefault("entity_id", entity_id_default)
    else:
        # No context available, use defaults
        user_id_default, entity_id_default = resolve_user_entity(None)
        data.setdefault("user_id", user_id_default)
        data.setdefault("entity_id", entity_id_default)
    
    # Ensure user_id and entity_id are set (already handled above)
    
    record = Interaction(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

# Get a single interaction by its id
def get_interaction_by_id(db: Session, id):
    return db.query(Interaction).filter(Interaction.id == id).first()

# List all interactions for a session, ordered by timestamp
def list_interactions_by_session(db: Session, session_id, n=None):
    query = db.query(Interaction).filter(Interaction.session_id == session_id).order_by(Interaction.timestamp)
    if n is not None:
        return query.limit(n).all()
    return query.all()

# Update an interaction (by id)
def update_interaction(db: Session, id, update_data: dict):
    record = db.query(Interaction).filter(Interaction.id == id).first()
    if not record:
        return None
    for key, value in update_data.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return record

# Delete an interaction (by id)
def delete_interaction(db: Session, id):
    record = db.query(Interaction).filter(Interaction.id == id).first()
    if not record:
        return False
    db.delete(record)
    db.commit()
    return True

# List all interactions for a session and agent, ordered by timestamp
def list_interactions_by_session_and_agent(db: Session, session_id, agent_id, n=None):
    query = db.query(Interaction).filter(
        Interaction.session_id == session_id,
        or_(Interaction.agent_id == agent_id, Interaction.agent_id == None)
    ).order_by(Interaction.timestamp)
    if n is not None:
        return query.limit(n).all()
    return query.all()

# List all interactions for a given agent, ordered by timestamp
def list_interactions_by_agent(db: Session, agent_id, n=None):
    query = db.query(Interaction).filter(Interaction.agent_id == agent_id).order_by(Interaction.timestamp)
    if n is not None:
        return query.limit(n).all()
    return query.all() 

def get_agent_sessions_and_interactions(db, agent_id, session_id=None):
    # Fetch all interactions for this agent, ordered by session and timestamp
    query = db.query(Interaction).filter(
        or_(
            Interaction.agent_id == agent_id,
            Interaction.agent_id == None
        )
    )
    if session_id:
        query = query.filter(Interaction.session_id == session_id)
    interactions = query.order_by(Interaction.session_id, Interaction.timestamp).all()

    # Group by session_id
    sessions_dict = defaultdict(list)
    for i in interactions:
        sessions_dict[str(i.session_id)].append(i)

    # Build the response structure (include user attachment metadata for multi-modal history)
    sessions = []
    for sid, session_interactions in sessions_dict.items():
        pairs = []
        user_msg = None
        user_attachments = None
        for msg in session_interactions:
            if msg.sender == "user":
                user_msg = msg.message
                user_attachments = getattr(msg, "attachments", None) if hasattr(msg, "attachments") else None
                if user_attachments is None:
                    user_attachments = []
            elif msg.sender == "assistant":
                pair = {
                    "user": user_msg or "",
                    "assistant": msg.message,
                    "user_attachments": user_attachments or [],
                }
                pair["user_message"] = pair["user"]
                pair["agent_response"] = pair["assistant"]
                pairs.append(pair)
                user_msg = None
                user_attachments = None

        if user_msg is not None:
            pair = {
                "user": user_msg,
                "assistant": None,
                "user_attachments": user_attachments or [],
            }
            pair["user_message"] = pair["user"]
            pair["agent_response"] = pair["assistant"]
            pairs.append(pair)

        sessions.append({
            "session_id": sid,
            "interactions": pairs,
        })

    return {
        "agent_id": str(agent_id),
        "sessions": sessions
    }