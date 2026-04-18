"""
Generic database utilities for agent operations.
Can be used across CrewAI, LangGraph, and other agent frameworks.
"""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from db.crud.agent import get_agent_by_id
from db.crud.interaction import get_agent_sessions_and_interactions
from db.models import Transaction, Session as SessionModel
from utils.logging_utils import get_logger

# Optional FastAPI import for HTTPException (only used in workflow context)
try:
    from fastapi import HTTPException
except ImportError:
    # Fallback for non-FastAPI contexts
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

logger = get_logger('database_utils')


class AgentDatabaseManager:
    """Manager class for agent database operations."""
    
    @staticmethod
    def get_agent_data(db: Session, agent_id: str) -> Dict[str, Any]:
        """
        Get agent data from database and convert to standardized format.
        
        Args:
            db: Database session
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing agent data
            
        Raises:
            ValueError: If agent is not found
        """
        agent_db = get_agent_by_id(db, agent_id)
        if not agent_db:
            raise ValueError(f"Agent with ID {agent_id} not found")
        
        return {
            'id': agent_id,
            'name': agent_db.name,
            'description': agent_db.description,
            'config': agent_db.config,
            'prompt_template': agent_db.prompt_template,
            'output_format': agent_db.output_format
        }
    
    @staticmethod
    def get_session_from_transaction(db: Session, transaction_id: uuid.UUID) -> str:
        """
        Get session ID from transaction.
        
        Args:
            db: Database session
            transaction_id: UUID of the transaction
            
        Returns:
            Session ID as string
            
        Raises:
            ValueError: If transaction is not found
        """
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not transaction:
            raise ValueError(f"Transaction with ID {transaction_id} not found")
        
        return str(transaction.session_id)
    
    @staticmethod
    def extract_agent_config_sections(agent_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Extract specific configuration sections from agent data.
        
        Args:
            agent_data: Agent data dictionary
            
        Returns:
            Dictionary with extracted config sections
        """
        config = agent_data.get('config', {})
        
        return {
            'mcp_config': config.get('mcp', {}),
            'memory_config': config.get('memory', {}),
            'llm_config': config.get('llms', {})
        }
    
    @staticmethod
    def ensure_session(db: Session, session_id: Optional[str], current_user=None) -> str:
        """
        Create or validate session_id; returns active session_id.
        
        Args:
            db: Database session
            session_id: Optional session ID to validate or None to create new
            current_user: Optional CurrentUser for auth context
            
        Returns:
            Active session_id as string
            
        Raises:
            HTTPException: If session_id is invalid or not found
        """
        from db.crud.session import create_session
        
        if not session_id or session_id == "None":
            session_id = str(uuid.uuid4())
            # Use CRUD function which handles user/entity context properly
            create_session(db, {
                "id": uuid.UUID(session_id),
                "start_time": datetime.now(),
                "status": "active"
            }, current_user=current_user)
            logger.info(f"Created new session: {session_id}")
            return session_id

        # validate existing
        try:
            session_uuid = uuid.UUID(session_id)
            existing_session = db.query(SessionModel).filter(SessionModel.id == session_uuid).first()
            if not existing_session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid session_id format: {session_id}")
        return session_id
    
    @staticmethod
    def create_transaction(db: Session, session_id: str, user_name: str, query: str, workflow_id: str, current_user=None) -> Transaction:
        """
        Create and commit a new transaction for workflow execution.
        
        Args:
            db: Database session
            session_id: Session ID
            user_name: User name
            query: User query/task
            workflow_id: Workflow agent ID
            current_user: Optional CurrentUser for auth context
            
        Returns:
            Created Transaction object
        """
        from db.crud.transaction import create_transaction as create_transaction_crud
        
        transaction_id = uuid.uuid4()
        current_time = datetime.now()
        # Use CRUD function which handles user/entity context properly
        transaction = create_transaction_crud(db, {
            "id": transaction_id,
            "session_id": uuid.UUID(session_id),
            "task_description": query,
            "status": "in_progress",
            "start_time": current_time,
            "input_data": {"query": query, "workflow_id": workflow_id, "user_name": user_name}
        }, current_user=current_user)
        return transaction
    
    @staticmethod
    def load_workflow_history(db: Session, workflow_id: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Load workflow-level chat history from database.
        
        Args:
            db: Database session
            workflow_id: Workflow agent ID
            session_id: Session ID
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of interaction dictionaries with 'user' and 'assistant' keys
        """
        db_chat_history: List[Dict[str, Any]] = []
        try:
            paired_interactions = get_agent_sessions_and_interactions(db, workflow_id, session_id)
            if paired_interactions.get("sessions") and len(paired_interactions["sessions"]) > 0:
                db_chat_history = paired_interactions["sessions"][0]["interactions"][-limit:]
        except Exception as e:
            logger.warning(f"Failed to retrieve chat_history from database: {e}")
        return db_chat_history
