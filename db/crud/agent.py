from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uuid
from db.models import (
    Agent,
    AgentExecutionLog,
    ToolExecutionLog,
    MemoryTransactionLog,
    RAGExecutionLog,
    LLMUsageLog,
    Interaction,
)

# Pydantic Models
class AgentSchema(BaseModel):
    """Base schema for Agent with common attributes"""
    name: str
    type: str
    description: str
    created_by_user_id: Optional[str] = None
    entity_id: Optional[str] = None
    llm_used: str
    prompt_template: Optional[str] = None
    config: Dict[str, Any]
    output_format: Optional[str] = None
    is_active: Optional[bool] = True

    model_config = ConfigDict(from_attributes=True)

class AgentCreate(AgentSchema):
    """Schema for creating a new agent"""
    pass

class AgentUpdate(BaseModel):
    """Schema for updating an agent"""
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    created_by_user_id: Optional[str] = None
    entity_id: Optional[str] = None
    llm_used: Optional[str] = None
    prompt_template: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    output_format: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class AgentInDB(AgentSchema):
    """Schema for agent in database (includes ID)"""
    id: str

    @classmethod
    def model_validate(cls, obj):
        # Convert UUID to string if necessary
        if hasattr(obj, 'id'):
            obj.id = str(obj.id)
        return super().model_validate(obj)

# CRUD Operations
def create_agent(
    db: Session, 
    data: AgentCreate, 
    current_user=None
) -> AgentInDB:
    """
    Create a new agent with user/entity context.
    
    Args:
        db: Database session
        data: AgentCreate schema
        current_user: Optional CurrentUser for auth context (if None, uses defaults)
        
    Security:
        Never trusts client-provided entity_id. Always derives from auth context.
    """
    from utils.user_context import enrich_data_with_user_context
    
    agent_data = data.model_dump()
    # Remove client-provided user/entity (security: never trust client)
    # Enrich with authenticated context
    agent_data = enrich_data_with_user_context(
        agent_data,
        current_user=current_user,
        user_id_field="created_by_user_id",
        include_user_id=True,
        include_entity_id=True
    )
    
    record = Agent(**agent_data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return AgentInDB.model_validate(record)

def _to_uuid(agent_id: str) -> Optional[uuid.UUID]:
    """Safely convert agent_id to UUID, returning None if invalid."""
    try:
        return uuid.UUID(str(agent_id))
    except (ValueError, TypeError):
        return None


def get_agent_by_id(db: Session, id: str) -> Optional[AgentInDB]:
    """Get an agent by ID. Returns None if ID is invalid or not found."""
    agent_uuid = _to_uuid(id)
    if not agent_uuid:
        return None
    record = db.query(Agent).filter(Agent.id == agent_uuid).first()
    return AgentInDB.model_validate(record) if record else None

def list_agents(db: Session) -> List[AgentInDB]:
    """List all agents"""
    records = db.query(Agent).all()
    return [AgentInDB.model_validate(record) for record in records]

def update_agent(
    db: Session, 
    id: str, 
    data: AgentUpdate,
    current_user=None
) -> Optional[AgentInDB]:
    """
    Update an agent with user/entity context validation.
    
    Args:
        db: Database session
        id: Agent ID
        data: AgentUpdate schema
        current_user: Optional CurrentUser for auth context
        
    Security:
        Removes any client-provided entity_id/user_id from update data.
        Ownership fields (created_by_user_id, entity_id) are not updated - they remain unchanged.
    """
    update_data = data.model_dump(exclude_unset=True)
    if not update_data:
        return None
    
    # Security: Remove ownership fields from update data (these should not be changed)
    # Agent model uses created_by_user_id (not user_id), and entity_id
    update_data.pop("entity_id", None)
    update_data.pop("user_id", None)
    update_data.pop("created_by_user_id", None)
    
    record = db.query(Agent).filter(Agent.id == id)
    record.update(update_data)
    db.commit()
    updated_record = record.first()
    return AgentInDB.model_validate(updated_record) if updated_record else None

def delete_agent(db: Session, id: str) -> bool:
    """Delete an agent after removing dependent records to satisfy FK constraints."""
    agent = db.query(Agent).filter(Agent.id == id).first()
    if not agent:
        return False

    try:
        # Ensure UUID type for comparisons
        agent_uuid = uuid.UUID(str(agent.id))

        # Delete dependent logs before deleting agent (avoid FK violations)
        db.query(AgentExecutionLog).filter(AgentExecutionLog.agent_id == agent_uuid).delete()
        db.query(ToolExecutionLog).filter(ToolExecutionLog.agent_id == agent_uuid).delete()
        db.query(MemoryTransactionLog).filter(MemoryTransactionLog.agent_id == agent_uuid).delete()
        db.query(RAGExecutionLog).filter(RAGExecutionLog.agent_id == agent_uuid).delete()
        db.query(LLMUsageLog).filter(LLMUsageLog.agent_id == agent_uuid).delete()
        db.query(Interaction).filter(Interaction.agent_id == agent_uuid).delete()

        # Finally delete the agent
        db.delete(agent)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise