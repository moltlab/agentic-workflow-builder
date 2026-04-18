"""
Generic agent logging and tracing functionality.
Can be used across CrewAI, LangGraph, and other agent frameworks.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from opentelemetry import trace

from db.models import ToolExecutionLog, Tool, Transaction
from db.crud.llm_usage import create_llm_usage_log
from db.crud.interaction import create_interaction
from utils.logging_utils import get_logger

logger = get_logger('agent_logging')

# Get OpenTelemetry tracer
tracer = trace.get_tracer("agent_logging")


def log_agent_step_original(agent_output, db: Session, transaction_id: uuid.UUID, 
                           agent_id: uuid.UUID, tool_usage: List[Dict], 
                           session_id: str = None):
    """
    Generic callback function to log agent-tool interactions to the database.
    Called after each step of an agent, regardless of framework.
    
    :param agent_output: The output from the agent's step
    :param db: SQLAlchemy database session
    :param transaction_id: UUID of the current transaction
    :param agent_id: UUID of the agent
    :param tool_usage: List of dictionaries to track tool usage information
    :param session_id: Session ID for tracking
    """
    from utils.user_context import get_user_context_from_transaction
    
    try:
        # Get user context from transaction
        try:
            user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
        except ValueError:
            # Transaction not found, use defaults
            from utils.user_context import resolve_user_entity
            user_id, entity_id = resolve_user_entity(None)
        
        # Check if the step involved a tool call
        if hasattr(agent_output, 'tool'):
            tool_name = agent_output.tool
            tool_input = agent_output.tool_input
            # CrewAI step may use .result or .output depending on version
            tool_output = getattr(agent_output, 'result', None) or getattr(agent_output, 'output', None)

            # Get tool_id from the database
            tool = db.query(Tool).filter(Tool.name == tool_name).first()
            if not tool:
                logger.warning(f"Tool {tool_name} not found in database")
                return
                
            # Create tool execution log entry
            tool_log = ToolExecutionLog(
                transaction_id=transaction_id,
                agent_id=agent_id,
                user_id=user_id,
                entity_id=entity_id,
                tool_id=tool.id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=0,
                input_data=tool_input,
                output_data=json.dumps(str(tool_output)),
                error_message=None
            )

            db.add(tool_log)
            db.commit()
            logger.info(f"Logged tool interaction: Agent used tool '{tool_name}'")
            
            # Add tool usage information to the list
            tool_usage.append({
                "tool_id": str(tool.id),
                "tool_name": tool_name,
                "start_time": tool_log.start_time,
                "end_time": tool_log.end_time,
                "duration_ms": tool_log.duration_ms,
                "input": tool_input,
                "output": str(tool_output)
            })
            
        elif hasattr(agent_output, 'is_error') and agent_output.is_error:
            # Log error case
            tool_log = ToolExecutionLog(
                transaction_id=transaction_id,
                agent_id=agent_id,
                user_id=user_id,
                entity_id=entity_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=0,
                input_data=json.dumps({"error": True}),
                output_data=None,
                error_message=str(agent_output.error)
            )
            
            db.add(tool_log)
            db.commit()
            logger.error(f"Logged error in agent step: {str(agent_output.error)}")
            
    except Exception as e:
        logger.error(f"Error logging to database: {e}")
        logger.exception("Full traceback:")
        db.rollback()

def log_agent_step_langgraph(tool_name: str, tool_input: str, tool_output: str, 
                           start_time: datetime, end_time: datetime,
                           db: Session, transaction_id: uuid.UUID, 
                           agent_id: uuid.UUID, tool_usage: List[Dict], 
                           session_id: str = None):
    """
    LangGraph-specific callback function to log tool executions to the database.
    Called from DBLogger callback handler with captured tool information.
    
    :param tool_name: Name of the tool that was executed
    :param tool_input: Input data passed to the tool
    :param tool_output: Output from the tool execution
    :param start_time: When tool execution started
    :param end_time: When tool execution ended
    :param db: SQLAlchemy database session
    :param transaction_id: UUID of the current transaction
    :param agent_id: UUID of the agent
    :param tool_usage: List of dictionaries to track tool usage information
    :param session_id: Session ID for tracking
    """
    from utils.user_context import get_user_context_from_transaction
    
    try:
        # Get user context from transaction
        try:
            user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
        except ValueError:
            # Transaction not found, use defaults
            from utils.user_context import resolve_user_entity
            user_id, entity_id = resolve_user_entity(None)
        
        # Get tool_id from the database
        tool = db.query(Tool).filter(Tool.name == tool_name).first()
        if not tool:
            logger.warning(f"Tool {tool_name} not found in database")
            return
        
        # Calculate duration
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Create tool execution log entry
        tool_log = ToolExecutionLog(
            transaction_id=transaction_id,
            agent_id=agent_id,
            user_id=user_id,
            entity_id=entity_id,
            tool_id=tool.id,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            input_data=tool_input,
            output_data=json.dumps(str(tool_output)),
            error_message=None
        )

        db.add(tool_log)
        db.commit()
        logger.info(f"Logged tool interaction: Agent used tool '{tool_name}' (duration: {duration_ms}ms)")
        
        # Add tool usage information to the list
        tool_usage.append({
            "tool_id": str(tool.id),
            "tool_name": tool_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "input": tool_input,
            "output": str(tool_output)
        })
        
    except Exception as e:
        logger.error(f"Error logging tool execution to database: {e}")
        logger.exception("Full traceback:")
        db.rollback()


# Using this function to log tool usage
def log_agent_step_with_tracing(agent_output, db: Session, transaction_id: uuid.UUID,
                                agent_id: uuid.UUID, tool_usage: List[Dict], 
                                session_id: str = None):
    """
    Enhanced agent step logging with OpenTelemetry tracing.
    Creates spans for dashboard visibility while logging to database.
    
    :param agent_output: The output from the agent's step
    :param db: SQLAlchemy database session
    :param transaction_id: UUID of the current transaction
    :param agent_id: UUID of the agent
    :param tool_usage: List of dictionaries to track tool usage information
    :param session_id: Session ID for tracking
    """
    with tracer.start_as_current_span("invoke_agent") as span:
        # Add span attributes for dashboard compatibility
        span.set_attribute("agent.id", str(agent_id))
        if session_id:
            span.set_attribute("session.id", session_id)
        
        # Add step metadata
        if hasattr(agent_output, 'agent'):
            span.set_attribute("gen_ai.agent.name", str(agent_output.agent))
        
        if hasattr(agent_output, 'task'):
            span.set_attribute("agent.step.task", str(agent_output.task))
        
        if hasattr(agent_output, 'input'):
            span.set_attribute("agent.step.input", str(agent_output.input))
        
        if hasattr(agent_output, 'output'):
            span.set_attribute("agent.step.output", str(agent_output.output))
        
        # Add step ID for tracking
        step_id = str(uuid.uuid4())
        span.set_attribute("agent.step.id", step_id)
        
        # Call the original logging function
        log_agent_step_original(agent_output, db, transaction_id, agent_id, tool_usage, session_id)
        
        logger.info(f"Logged agent step with trace span: {hex(span.get_span_context().span_id)}")


def create_agent_execution_span(span_name: str, agent_name: str, agent_id: str, 
                               session_id: str, query: str, tools: List = None) -> Dict[str, Any]:
    """
    Create a standardized OpenTelemetry span for agent execution.
    Returns span attributes for consistency across frameworks.
    
    :param span_name: Name of the span (e.g., "crew-kickoff", "langgraph-agent-kickoff")
    :param agent_name: Name of the agent
    :param agent_id: ID of the agent
    :param session_id: Session ID
    :param query: User query/task
    :param tools: List of available tools
    :return: Dictionary of span attributes
    """
    span_attributes = {
        "gen_ai.agent.name": agent_name,
        "agent.id": agent_id,
        "session.id": session_id,
        "gen_ai.prompt": f"Current Task: {query}\n\nAgent: {agent_name}\nSession: {session_id}",
        "query.length": len(query)
    }
    
    # Add tools information if available
    if tools:
        tools_list = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
        span_attributes["gen_ai.agent.tools"] = ", ".join(tools_list)
    
    return span_attributes


def log_llm_usage(db: Session, transaction_id: uuid.UUID, agent_id: uuid.UUID,
                 llm_metadata: Dict[str, Any], query: str, result: str,
                 start_time: datetime, end_time: datetime) -> None:
    """
    Generic function to log LLM usage across different frameworks.
    
    :param db: Database session
    :param transaction_id: Transaction ID
    :param agent_id: Agent ID
    :param llm_metadata: LLM metadata from LLMFactory
    :param query: Input query
    :param result: LLM result
    :param start_time: Execution start time
    :param end_time: Execution end time
    """
    from utils.user_context import get_user_context_from_transaction
    
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    
    # Get user context from transaction
    try:
        user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
    except ValueError:
        # Transaction not found, will use defaults in create_llm_usage_log
        user_id, entity_id = None, None
    
    create_llm_usage_log(db, {
        "transaction_id": transaction_id,
        "agent_id": agent_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "model_name": llm_metadata.get("model_name"),
        "model_provider": llm_metadata.get("model_provider"),
        "temperature": llm_metadata.get("temperature"),
        "max_tokens": llm_metadata.get("max_tokens"),
        "total_tokens_used": llm_metadata.get("total_tokens", 0),
        "response_latency_ms": llm_metadata.get("latency_ms", 0),
        "input_data": {"query": query},
        "output_data": {"result": str(result)}
    }, user_id=user_id, entity_id=entity_id)


def log_agent_interactions(
    db: Session,
    session_id: str,
    agent_id: uuid.UUID,
    query: str,
    result: str,
    transaction_id: Optional[uuid.UUID] = None,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Generic function to log agent interactions (user query + agent response).
    Optionally store attachment metadata (media_id, media_type, cloud_path) on the user message.

    :param attachments: Optional list of {"media_id", "media_type", "cloud_path"} for the user message.
    """
    from utils.user_context import get_user_context_from_transaction

    if not user_id or not entity_id:
        if transaction_id:
            try:
                user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
            except ValueError:
                pass

    user_data: Dict[str, Any] = {
        "session_id": session_id,
        "sender": "user",
        "message": query,
        "timestamp": datetime.now(),
        "agent_id": agent_id,
        "message_metadata": {},
    }
    if attachments:
        user_data["attachments"] = attachments

    create_interaction(db, user_data, user_id=user_id, entity_id=entity_id)

    create_interaction(db, {
        "session_id": session_id,
        "sender": "assistant",
        "message": str(result),
        "timestamp": datetime.now(),
        "agent_id": agent_id,
        "message_metadata": {},
    }, user_id=user_id, entity_id=entity_id)


def serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def serialize_tool_usage(tool_usage: List[Dict]) -> List[Dict]:
    """
    Serialize tool usage data for JSON compatibility.
    
    :param tool_usage: List of tool usage dictionaries
    :return: Serialized tool usage data
    """
    serialized_tool_usage = []
    for tool in tool_usage:
        serialized_tool = {
            "tool_id": tool["tool_id"],
            "tool_name": tool["tool_name"],
            "start_time": serialize_datetime(tool["start_time"]),
            "end_time": serialize_datetime(tool["end_time"]),
            "duration_ms": tool["duration_ms"],
            "input": tool["input"],
            "output": tool["output"]
        }
        serialized_tool_usage.append(serialized_tool)
    return serialized_tool_usage
