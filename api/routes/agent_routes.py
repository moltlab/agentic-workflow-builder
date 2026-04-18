from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Any, Dict
import time
import uuid
import json
import os
from db.models import Agent, Transaction, AgentExecutionLog, ToolExecutionLog, LLMUsageLog, Session as SessionModel
from db.config import SessionLocal, get_db
from sqlalchemy.orm import Session
from db.crud.agent import (
    AgentCreate,
    AgentUpdate,
    AgentInDB,
    create_agent,
    list_agents,
    get_agent_by_id,
    update_agent,
    delete_agent
)
from db.crud.interaction import get_agent_sessions_and_interactions
from db.crud.tool import list_tools
from db.crud.transaction import get_recent_transactions as get_recent_transactions_pg, get_session_stats as get_session_stats_pg
from api.schemas.requests import (
    QueryRequest,
    DocumentRequest,
    HistoryRequest,
    TransactionResponse,
    ErrorResponse
)
from agents.langgraph_agent_v2 import get_and_run_langgraph_agent_v2
from agents.base_agent import BaseAgent
from agents.shared import build_multimodal_content
from agents.shared.datasource_scope import validate_agent_mcp_datasource_bindings
from agents.shared.agent_logging import log_agent_interactions
from utils.user_context import get_user_context_from_transaction
from pydantic import BaseModel, ValidationError
from datetime import datetime
from utils.logging_utils import get_logger
import sqlite3
import logging

# Optional database imports - handle missing dependencies gracefully
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False



try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)

agent_router = APIRouter()

# ---------------------------------------------------------------------------
# Media helpers (multi-modal support)
# ---------------------------------------------------------------------------
from utils.media_storage import GCSMediaStorage, MediaItem

_gcs_storage_instance: Optional[GCSMediaStorage] = None


def _get_gcs_storage() -> Optional[GCSMediaStorage]:
    """Return GCS client if configured; None otherwise (graceful degradation)."""
    global _gcs_storage_instance
    if _gcs_storage_instance is not None:
        return _gcs_storage_instance
    try:
        _gcs_storage_instance = GCSMediaStorage()
        return _gcs_storage_instance
    except ValueError:
        return None


def resolve_media_ids(
    media_ids: Optional[List[str]],
    db: Optional[Session] = None,
    session_id: Optional[str] = None,
) -> Optional[List[MediaItem]]:
    """
    Convert a list of media_ids into MediaItem objects with signed GCS URLs.

    - If each entry is a valid UUID string: look up Media by id from DB (optional session_id scope).
    - Legacy format "cloud_path::media_type" is still supported when DB lookup is not used.
    """
    if not media_ids:
        logger.info("[multimodal] resolve_media_ids: no media_ids, returning None")
        return None

    storage = _get_gcs_storage()
    if storage is None:
        logger.warning("media_ids provided but GCS storage is not configured — ignoring")
        return None

    from db.crud.media import get_media_by_ids

    items: List[MediaItem] = []
    uuid_ids: List[uuid.UUID] = []
    legacy_entries: List[tuple] = []  # (cloud_path, media_type)

    for entry in media_ids:
        if not entry or not isinstance(entry, str):
            continue
        entry = entry.strip()
        try:
            uid = uuid.UUID(entry)
            uuid_ids.append(uid)
        except ValueError:
            if "::" in entry:
                path_part, media_type = entry.rsplit("::", 1)
                legacy_entries.append((path_part.strip(), media_type.strip()))
            else:
                logger.warning(f"Skipping malformed media_id: {entry}")

    logger.info(
        "[multimodal] resolve_media_ids: parsed uuid_count=%s legacy_count=%s session_id=%s db=%s",
        len(uuid_ids), len(legacy_entries), session_id, db is not None,
    )

    if db and uuid_ids:
        session_uuid = uuid.UUID(session_id) if session_id else None
        media_rows = get_media_by_ids(db, uuid_ids, session_id=session_uuid)
        logger.info(
            "[multimodal] resolve_media_ids: get_media_by_ids requested %s ids with session_id=%s -> found %s rows",
            len(uuid_ids), session_uuid, len(media_rows),
        )
        for row in media_rows:
            signed_url = storage.get_signed_url(row.cloud_path)
            items.append(
                MediaItem(
                    media_id=str(row.id),
                    filename=row.filename,
                    media_type=row.media_type,
                    cloud_path=row.cloud_path,
                    file_size=row.file_size,
                    signed_url=signed_url,
                )
            )

    for cloud_path, media_type in legacy_entries:
        signed_url = storage.get_signed_url(cloud_path)
        items.append(
            MediaItem(
                media_id=cloud_path.split("/")[-2] if "/" in cloud_path else "legacy",
                filename=cloud_path.split("/")[-1] if "/" in cloud_path else "unknown",
                media_type=media_type,
                cloud_path=cloud_path,
                file_size=0,
                signed_url=signed_url,
            )
        )

    return items if items else None


# Helper classes and functions
def _serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

async def log_transaction(
    db: Session,
    query: QueryRequest,
    result: str,
    agent_id: str,
    current_user=None,
    media_ids: Optional[List[str]] = None,
) -> str:
    """
    Log the transaction and its associated logs in the database.
    Returns the transaction ID.
    When media_ids (UUID strings from Media table) are provided, stores them in
    input_data and links Media rows to this transaction via transaction_id.
    """
    from utils.user_context import resolve_user_entity

    try:
        current_time = datetime.now()
        user_id, entity_id = resolve_user_entity(current_user)

        # Handle session_id - convert string to UUID if needed
        session_id = None
        if query.session_id:
            try:
                session_id = uuid.UUID(query.session_id)
                existing_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                if not existing_session:
                    from db.crud.session import create_session
                    create_session(db, {
                        "id": session_id,
                        "start_time": current_time,
                        "status": "active"
                    }, current_user=current_user)
                    logger.info(f"Created new session with ID: {session_id}")
            except ValueError:
                session_id = uuid.uuid4()
                from db.crud.session import create_session
                create_session(db, {
                    "id": session_id,
                    "start_time": current_time,
                    "status": "active"
                }, current_user=current_user)
                logger.info(f"Generated new session with ID: {session_id}")

        input_data: Dict[str, Any] = {"query": query.query, "agent": query.agent}
        if media_ids:
            input_data["media_ids"] = media_ids

        # Create main transaction
        from db.crud.transaction import create_transaction
        transaction = create_transaction(db, {
            "id": uuid.uuid4(),
            "task_description": query.query,
            "status": "completed",
            "start_time": current_time,
            "end_time": current_time,
            "input_data": input_data,
            "final_output": {"result": result},
            "session_id": session_id
        }, current_user=current_user)

        # Link Media rows to this transaction (only valid UUIDs from Media table)
        if media_ids:
            from db.crud.media import update_media_transaction_id
            valid_uuids: List[uuid.UUID] = []
            for m in media_ids:
                if not m or not isinstance(m, str):
                    continue
                try:
                    valid_uuids.append(uuid.UUID(m))
                except ValueError:
                    pass
            if valid_uuids:
                update_media_transaction_id(db, valid_uuids, transaction.id)

        # Create agent execution log
        # Convert agent_id to UUID safely (agent_id should already be validated as existing)
        if isinstance(agent_id, str):
            agent_uuid = uuid.UUID(agent_id)
        elif isinstance(agent_id, uuid.UUID):
            agent_uuid = agent_id
        else:
            # Fallback: try to convert to string first
            agent_uuid = uuid.UUID(str(agent_id))
        
        agent_log = AgentExecutionLog(
            transaction_id=transaction.id,
            agent_id=agent_uuid,
            user_id=user_id,
            entity_id=entity_id,
            start_time=current_time,
            end_time=current_time,
            input_data={"query": query.query},
            output_data={"result": result}
        )
        db.add(agent_log)

        # If the result contains tool usage information, log it
        if isinstance(result, dict) and "tool_usage" in result:
            for tool_usage in result["tool_usage"]:
                # Convert datetime objects to ISO format strings
                tool_usage = {k: _serialize_datetime(v) for k, v in tool_usage.items()}
                # Convert agent_id to UUID safely
                if isinstance(agent_id, str):
                    agent_uuid = uuid.UUID(agent_id)
                elif isinstance(agent_id, uuid.UUID):
                    agent_uuid = agent_id
                else:
                    agent_uuid = uuid.UUID(str(agent_id))
                
                tool_log = ToolExecutionLog(
                    transaction_id=transaction.id,
                    agent_id=agent_uuid,
                    user_id=user_id,
                    entity_id=entity_id,
                    start_time=tool_usage.get("start_time", current_time),
                    end_time=tool_usage.get("end_time", current_time),
                    tool_id=uuid.UUID(tool_usage["tool_id"]) if "tool_id" in tool_usage else None,
                    input_data=tool_usage.get("input", {}),
                    output_data=tool_usage.get("output", {})
                )
                db.add(tool_log)

        # If the result contains LLM usage information, log it
        if isinstance(result, dict) and "llm_usage" in result:
            llm_usage = result["llm_usage"]
            # Convert datetime objects to ISO format strings
            llm_usage = {k: _serialize_datetime(v) for k, v in llm_usage.items()}
            # Convert agent_id to UUID safely
            if isinstance(agent_id, str):
                agent_uuid = uuid.UUID(agent_id)
            elif isinstance(agent_id, uuid.UUID):
                agent_uuid = agent_id
            else:
                agent_uuid = uuid.UUID(str(agent_id))
            
            llm_log = LLMUsageLog(
                transaction_id=transaction.id,
                agent_id=agent_uuid,
                user_id=user_id,
                entity_id=entity_id,
                start_time=llm_usage.get("start_time", current_time),
                end_time=llm_usage.get("end_time", current_time),
                model_name=llm_usage.get("model_name"),
                model_provider=llm_usage.get("model_provider"),
                temperature=llm_usage.get("temperature"),
                max_tokens=llm_usage.get("max_tokens"),
                total_tokens_used=llm_usage.get("total_tokens"),
                response_latency_ms=llm_usage.get("latency_ms"),
                prompt=llm_usage.get("prompt"),
                response=llm_usage.get("response")
            )
            db.add(llm_log)

        db.commit()
        return str(transaction.id)

    except Exception as e:
        db.rollback()
        logger.error(f"Error logging transaction: {str(e)}")
        raise

async def get_database_summary(db_type: str, connection_string: str = None, host: str = None, port: int = None, database: str = None, username: str = None, password: str = None) -> dict:
    """Get database summary including table count and table names."""
    try:
        tables = []
        
        if connection_string:
            # Use connection string
            if db_type == "sqlite":
                conn = sqlite3.connect(connection_string)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
            else:
                if not SQLALCHEMY_AVAILABLE:
                    return {"error": "SQLAlchemy not available for database summary"}
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    if db_type == "postgresql":
                        result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
                    elif db_type == "mysql":
                        result = conn.execute(text("SHOW TABLES"))
                    else:
                        return {"error": f"Unsupported database type for summary: {db_type}"}
                    
                    tables = [row[0] for row in result.fetchall()]
                engine.dispose()
        else:
            # Use individual parameters
            if db_type == "sqlite":
                conn = sqlite3.connect(database)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                
            elif db_type == "postgresql":
                if not PSYCOPG2_AVAILABLE:
                    return {"error": "PostgreSQL support not available"}
                conn = psycopg2.connect(
                    host=host,
                    port=port or 5432,
                    database=database,
                    user=username,
                    password=password
                )
                cursor = conn.cursor()
                cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                
            elif db_type == "mysql":
                if not PYMYSQL_AVAILABLE:
                    return {"error": "MySQL support not available"}
                conn = pymysql.connect(
                    host=host,
                    port=port or 3306,
                    database=database,
                    user=username,
                    password=password
                )
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
        
        return {
            "total_tables": len(tables),
            "table_names": tables,
            "database_type": db_type
        }
        
    except Exception as e:
        logger.error(f"Error getting database summary: {str(e)}")
        return {"error": f"Failed to get database summary: {str(e)}"}

# Add dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@agent_router.post("/create-agent", response_model=AgentInDB, status_code=201)
async def create_new_agent(
    agent_data: AgentCreate,
    db: Session = Depends(get_db),
):
    """Create a new agent with the specified configuration"""
    try:
        # Validate required fields
        if not agent_data.name or not agent_data.name.strip():
            raise ValueError("Agent name is required")
        if not agent_data.description or not agent_data.description.strip():
            raise ValueError("Agent description is required")
        if not agent_data.type:
            raise ValueError("Agent type is required")
        if not agent_data.llm_used:
            raise ValueError("LLM model is required")
        if not isinstance(agent_data.config, dict):
            raise ValueError("Config must be a valid JSON object")

        validate_agent_mcp_datasource_bindings(agent_data.config)

        # Create the agent (user/entity context handled in create_agent)
        # Security: Never trust client-provided entity_id
        new_agent = create_agent(db, agent_data)
        return new_agent
        
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        # Log the error here if you have logging set up
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while creating agent: {str(e)}"
        )

@agent_router.get("/agents", response_model=List[AgentInDB])
async def get_agents(
    db: Session = Depends(get_db),
):
    """Get all agents"""
    return list_agents(db)

@agent_router.get("/agents/{agent_id}", response_model=AgentInDB)
async def get_agent(
    agent_id: str,
    db: Session = Depends(get_db),
):
    """Get a specific agent by ID"""
    agent = get_agent_by_id(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@agent_router.put("/agents/{agent_id}", response_model=AgentInDB)
async def update_agent_endpoint(
    agent_id: str,
    agent_data: AgentUpdate,
    db: Session = Depends(get_db),
):
    """Update an agent"""
    try:
        if agent_data.config is not None:
            validate_agent_mcp_datasource_bindings(agent_data.config)
        updated_agent = update_agent(db, agent_id, agent_data)
        if not updated_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return updated_agent
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@agent_router.delete("/agents/{agent_id}")
async def delete_agent_endpoint(
    agent_id: str,
    db: Session = Depends(get_db),
):
    """Delete an agent"""
    if not delete_agent(db, agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}


@agent_router.get("/tools")
async def get_tools(
    db: Session = Depends(get_db),
):
    """
    List all tools from the database (includes MCP-synced tools).
    Returns id, name, description so Tool.name can be used where tool_execution_log
    must resolve tool_id (e.g. hierarchical workflow tool_node wizard).
    """
    try:
        tools = list_tools(db)
        return JSONResponse([
            {"id": str(t.id), "name": t.name, "description": (t.description or "")}
            for t in tools
        ])
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@agent_router.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    limit: int = 10,
    agent: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Show command history from the database (PostgreSQL)."""
    transactions = get_recent_transactions_pg(db, limit=limit, agent_name=agent, session_id=session_id)
    if not transactions:
        return {"message": "No transactions found"}
    out = []
    for tx in transactions:
        resp = tx.get("response")
        if resp is not None:
            resp = str(resp)
            resp = (resp[:100] + "...") if len(resp) > 100 else resp
        out.append({
            "transaction_id": tx["transaction_id"],
            "agent_name": tx.get("agent_name") or "Not specified",
            "timestamp": tx.get("timestamp"),
            "status": tx.get("status", ""),
            "input": tx.get("input", ""),
            "duration_ms": tx.get("duration_ms"),
            "response": resp,
        })
    return {"transactions": out}

@agent_router.get("/session-stats")
async def get_session_stats(
    session_id: str,
    db: Session = Depends(get_db),
):
    """Show statistics for a specific session (PostgreSQL)."""
    return get_session_stats_pg(db, session_id)

@agent_router.post("/test-database-connection")
async def test_database_connection(
    request: Request,
):
    """Test database connection with provided credentials"""
    try:
        body = await request.json()
        
        # Validate required fields
        if not body.get("db_type"):
            raise HTTPException(status_code=400, detail="Database type is required")
        
        db_type = body["db_type"]
        connection_string = body.get("connection_string")
        
        # If connection string is provided, use it directly
        if connection_string:
            try:
                if db_type == "sqlite":
                    # Test SQLite connection
                    conn = sqlite3.connect(connection_string)
                    conn.execute("SELECT 1")
                    conn.close()
                else:
                    if not SQLALCHEMY_AVAILABLE:
                        raise HTTPException(status_code=400, detail="SQLAlchemy not available. Please install sqlalchemy: pip install sqlalchemy")
                    # Test other database types using SQLAlchemy
                    engine = create_engine(connection_string)
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    engine.dispose()
                
                # Get database summary
                summary = await get_database_summary(db_type, connection_string, None, None, None, None, None)
                
                return {
                    "status": "success", 
                    "message": "Database connection successful",
                    "summary": summary
                }
                
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")
        
        # Otherwise, use individual connection parameters
        host = body.get("host")
        port = body.get("port")
        database = body.get("database")
        username = body.get("username")
        password = body.get("password")
        
        # Validate required fields based on database type
        if db_type == "sqlite":
            if not database:
                raise HTTPException(status_code=400, detail="Database name is required for SQLite")
        else:
            if not all([host, database, username]):
                raise HTTPException(status_code=400, detail="Host, database name, and username are required")
        
        try:
            if db_type == "sqlite":
                # SQLite doesn't need host/port/username/password
                conn = sqlite3.connect(database)
                conn.execute("SELECT 1")
                conn.close()
                
            elif db_type == "postgresql":
                if not PSYCOPG2_AVAILABLE:
                    raise HTTPException(status_code=400, detail="PostgreSQL support not available. Please install psycopg2: pip install psycopg2-binary")
                # Test PostgreSQL connection
                conn = psycopg2.connect(
                    host=host,
                    port=port or 5432,
                    database=database,
                    user=username,
                    password=password
                )
                conn.close()
                
            elif db_type == "mysql":
                if not PYMYSQL_AVAILABLE:
                    raise HTTPException(status_code=400, detail="MySQL support not available. Please install pymysql: pip install pymysql")
                # Test MySQL connection
                conn = pymysql.connect(
                    host=host,
                    port=port or 3306,
                    database=database,
                    user=username,
                    password=password
                )
                conn.close()
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported database type: {db_type}")
            
            # Get database summary
            summary = await get_database_summary(db_type, None, host, port, database, username, password)
            
            return {
                "status": "success", 
                "message": "Database connection successful",
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@agent_router.post("/ask-agent-stream")
async def ask_agent_stream(
    request: Request,
    db: Session = Depends(get_db),
):
    """Stream LLM response from a LangGraph agent."""
    try:
        # Parse the request body manually
        body = await request.json()
        
        # Validate required fields
        if not body.get("query"):
            raise HTTPException(status_code=400, detail="Query is required")
        if not body.get("agent"):
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        fw = (body.get("framework") or "langgraph").lower()
        if fw != "langgraph":
            raise HTTPException(status_code=400, detail="Only 'langgraph' is supported")

        # Resolve media attachments (multi-modal)
        media_items = resolve_media_ids(body.get("media_ids"), db=db, session_id=body.get("session_id"))
        
        # Handle session_id the same way as /ask API: create if missing; if client sent one, ensure it exists
        if body.get("session_id") is None:
            from db.crud.session import create_session
            session_id = uuid.uuid4()
            session = create_session(db, {
                "id": session_id,
                "start_time": datetime.now(),
                "status": "active"
            })
            body["session_id"] = session_id
        else:
            try:
                session_id = uuid.UUID(body.get("session_id"))
                existing_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                if not existing_session:
                    from db.crud.session import create_session
                    create_session(db, {
                        "id": session_id,
                        "start_time": datetime.now(),
                        "status": "active"
                    })
                    logger.info(f"Created new session with client-provided ID (stream): {session_id}")
            except (ValueError, TypeError):
                from db.crud.session import create_session
                session_id = uuid.uuid4()
                create_session(db, {
                    "id": session_id,
                    "start_time": datetime.now(),
                    "status": "active"
                })
                body["session_id"] = str(session_id)
                logger.info(f"Invalid client session_id (stream); generated new session: {session_id}")
        
        # Validate agent exists
        try:
            if body["agent"]:
                try:
                    uuid.UUID(body["agent"])
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid agent ID format: {body['agent']}"
                    )
            
            agent = get_agent_by_id(db, body["agent"])
            if not agent:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent with ID {body['agent']} not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent ID: {body['agent']}"
            )
        
        user_permissions = None
        
        # Create transaction for logging
        query_request = QueryRequest(
            query=body["query"],
            agent=body["agent"],
            session_id=str(session_id) if session_id else None,
            framework="langgraph",
            media_ids=body.get("media_ids"),
        )
        transaction_id = await log_transaction(
            db, query_request, None, body["agent"],
            media_ids=body.get("media_ids"),
        )
        
        # Create streaming response generator
        async def stream_generator():
            def persist_stream_result(result_str: str) -> None:
                """Persist interactions (with attachments), transaction final_output, and agent execution log (sync with /ask)."""
                attachments = None
                if media_items:
                    attachments = [
                        {"media_id": getattr(m, "media_id", None), "media_type": getattr(m, "media_type", None), "cloud_path": getattr(m, "cloud_path", None)}
                        for m in media_items
                    ]
                    attachments = [a for a in attachments if a.get("media_id") and a.get("cloud_path")] or None
                try:
                    uid, eid = get_user_context_from_transaction(db, uuid.UUID(transaction_id))
                except Exception:
                    uid, eid = None, None
                log_agent_interactions(
                    db, str(session_id), uuid.UUID(body["agent"]), body["query"], result_str,
                    transaction_id=uuid.UUID(transaction_id), user_id=uid, entity_id=eid,
                    attachments=attachments,
                )
                t = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
                if t:
                    t.final_output = {"result": result_str}
                    t.status = "completed"
                    t.end_time = datetime.now()
                    db.commit()
                ael = db.query(AgentExecutionLog).filter(AgentExecutionLog.transaction_id == uuid.UUID(transaction_id)).first()
                if ael:
                    ael.end_time = datetime.now()
                    ael.output_data = {"result": result_str}
                    db.commit()

            def persist_stream_error(error_message: str) -> None:
                """On stream error: update transaction and agent execution log to failed (sync with /ask)."""
                t = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
                if t:
                    t.status = "failed"
                    t.end_time = datetime.now()
                    t.final_output = {"error": error_message}
                    db.commit()
                ael = db.query(AgentExecutionLog).filter(AgentExecutionLog.transaction_id == uuid.UUID(transaction_id)).first()
                if ael:
                    ael.end_time = datetime.now()
                    ael.output_data = {"error": error_message}
                    db.commit()

            try:
                # Send session_id first so frontend can use it if it did not send one
                yield f"data: {json.dumps({'status': 'session', 'session_id': str(session_id)})}\n\n"
                import asyncio
                from langchain.callbacks.base import AsyncCallbackHandler
                from langchain_core.messages import HumanMessage, AIMessage
                from datetime import datetime
                
                # Custom callback handler to capture streaming tokens
                class StreamingCallbackHandler(AsyncCallbackHandler):
                    def __init__(self, queue: asyncio.Queue):
                        self.queue = queue

                    async def on_chat_model_start(self, serialized, messages, **kwargs):
                        """Called when chat model starts. Required to avoid NotImplementedError warning."""
                        pass

                    async def on_llm_new_token(self, token: str, **kwargs):
                        await self.queue.put({"type": "token", "content": token})

                    async def on_llm_end(self, response, **kwargs):
                        await self.queue.put({"type": "end", "content": "[[END]]"})

                    async def on_agent_action(self, action, **kwargs):
                        await self.queue.put({"type": "agent_action", "content": str(action)})

                    async def on_agent_finish(self, finish, **kwargs):
                        await self.queue.put({"type": "agent_finish", "content": str(finish)})

                # Create queue for streaming tokens
                queue: asyncio.Queue = asyncio.Queue()
                handler = StreamingCallbackHandler(queue)
                
                # LangGraph streaming (astream_events)
                from agents.langgraph_agent_v2 import create_langgraph_agent_from_config

                # Create LangGraph agent workflow with streaming enabled
                app, metadata = await create_langgraph_agent_from_config(
                    db, body["agent"], body["query"], uuid.UUID(transaction_id),
                    include_memory=True, streaming=True, user_permissions=user_permissions
                )

                # Convert chat history to LangChain messages (include user_attachments when present)
                chat_history = metadata.get('chat_history', [])
                langchain_messages = []
                from utils.media_storage import get_media_storage
                from agents.shared.message_builder import build_content as build_multimodal_content, media_items_from_attachment_metadata
                storage = get_media_storage()
                for interaction in chat_history:
                    user_text = interaction.get('user_message') or interaction.get('user')
                    if user_text:
                        user_attachments = interaction.get('user_attachments') or []
                        if user_attachments and storage:
                            history_media = media_items_from_attachment_metadata(storage, user_attachments)
                            content = build_multimodal_content(user_text, history_media)
                        else:
                            content = user_text
                        langchain_messages.append(HumanMessage(content=content))
                    if interaction.get('agent_response') or interaction.get('assistant'):
                        assistant_text = interaction.get('agent_response') or interaction.get('assistant')
                        langchain_messages.append(AIMessage(content=assistant_text))

                # Add current query (and optional multi-modal content)
                stream_content = build_multimodal_content(body["query"], media_items)
                langchain_messages.append(HumanMessage(content=stream_content))

                final_result = ""
                try:
                    async for event in app.astream_events(
                        {"messages": langchain_messages},
                        version="v2"
                    ):
                        event_type = event.get("event")

                        if event_type == "on_chat_model_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, 'content') and chunk.content:
                                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                                final_result += chunk.content

                        elif event_type == "on_tool_start":
                            tool_name = event.get("name", "unknown")
                            yield f"data: {json.dumps({'status': 'tool_start', 'content': f'Calling tool: {tool_name}'})}\n\n"

                        elif event_type == "on_tool_end":
                            tool_name = event.get("name", "unknown")
                            yield f"data: {json.dumps({'status': 'tool_end', 'content': f'Tool completed: {tool_name}'})}\n\n"

                    result_str = final_result or "Response completed"
                    yield f"data: {json.dumps({'status': 'completed', 'result': result_str})}\n\n"
                    persist_stream_result(result_str)

                except Exception as e:
                    logger.error(f"LangGraph streaming error: {e}")
                    yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                    persist_stream_error(str(e))

            except Exception as e:
                # Send error signal and persist failed state
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                try:
                    persist_stream_error(str(e))
                except Exception:
                    pass
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@agent_router.post("/ask-agent")
@agent_router.post("/ask")  # Backward compatibility alias
async def ask_agent(
    request: Request,
):
    db = next(get_db())
    try:
        # Parse the request body manually
        body = await request.json()
        logger.info("Request body: %s", body)

        # Validate required fields
        if not body.get("query"):
            raise HTTPException(status_code=400, detail="Query is required")
        if not body.get("agent"):
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        fw = (body.get("framework") or "langgraph").lower()
        if fw != "langgraph":
            raise HTTPException(status_code=400, detail="Only 'langgraph' is supported")

        # Capture client session_id before we may create one (for media lookup: only filter by session when client sent it)
        client_sent_session_id = body.get("session_id")

        if client_sent_session_id is None:
            from db.crud.session import create_session
            session_id = uuid.uuid4()
            session = create_session(db, {
                "id": session_id,
                "start_time": datetime.now(),
                "status": "active"
            })
            body["session_id"] = session_id
        else:
            try:
                session_id = uuid.UUID(client_sent_session_id)
                existing_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                if not existing_session:
                    from db.crud.session import create_session
                    create_session(db, {
                        "id": session_id,
                        "start_time": datetime.now(),
                        "status": "active"
                    })
                    logger.info(f"Created new session with client-provided ID: {session_id}")
            except (ValueError, TypeError):
                from db.crud.session import create_session
                session_id = uuid.uuid4()
                create_session(db, {
                    "id": session_id,
                    "start_time": datetime.now(),
                    "status": "active"
                })
                body["session_id"] = str(session_id)
                logger.info(f"Invalid client session_id; generated new session: {session_id}")

        # Resolve media attachments (multi-modal). Only filter by session_id when client sent one,
        # so uploads without session (or from another session) are still found by media_id.
        media_items = resolve_media_ids(
            body.get("media_ids"),
            db=db,
            session_id=str(client_sent_session_id) if client_sent_session_id is not None else None,
        )
        logger.info(
            "[multimodal] /ask media_ids=%s -> resolved media_items count=%s (session_filter=%s)",
            body.get("media_ids"),
            len(media_items) if media_items else 0,
            client_sent_session_id is not None,
        )
        if media_items:
            for i, m in enumerate(media_items):
                logger.info(
                    "[multimodal] media_items[%s] id=%s type=%s has_signed_url=%s url_prefix=%s",
                    i, getattr(m, "media_id", None), getattr(m, "media_type", None),
                    bool(getattr(m, "signed_url", None)),
                    (getattr(m, "signed_url", "") or "")[:60],
                )

        # Create QueryRequest instance with framework
        # Use model_validate to avoid FastAPI intercepting ValidationError
        try:
            query = QueryRequest.model_validate({
                "query": body["query"],
                "agent": body["agent"],
                "session_id": str(session_id) if session_id else None,
                "framework": "langgraph",
                "media_ids": body.get("media_ids"),
            })
        except ValidationError as e:
            # Handle Pydantic validation errors - convert to 400 instead of 422
            error_details = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request data: {error_details}"
            )
        except Exception as e:
            # Handle any other errors
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request data: {str(e)}"
            )
        
        result = None

        # Validate agent exists if it's a database agent
        try:
            # First check if agent exists (before validating UUID format)
            agent = get_agent_by_id(db, query.agent)
            if not agent:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent with ID {query.agent} not found"
                )
        except HTTPException:
            # Re-raise HTTP exceptions (including 404)
            raise
        except Exception as e:
            # If database query fails (e.g., invalid UUID format), treat as not found
            # This allows non-UUID strings to be checked and return 404 instead of 400
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {query.agent} not found"
            )

        user_permissions = None
        
        # Create transaction first (include media_ids for DB linking and audit)
        transaction_id = await log_transaction(
            db, query, None, query.agent, media_ids=query.media_ids
        )

        try:
            # Handle database agents with selected framework
            logger.info(
                "[multimodal] invoking agent framework=%s with media_items count=%s",
                query.framework,
                len(media_items) if media_items else 0,
            )
            result = await get_and_run_langgraph_agent_v2(
                db, query.agent, query.query, uuid.UUID(transaction_id),
                user_permissions=user_permissions, media_items=media_items
            )
            
            if result is None:
                raise HTTPException(status_code=500, detail="Agent returned no result")
            
            # Update the transaction with the result
            transaction = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
            if transaction:
                transaction.final_output = result
                transaction.status = "completed"
                transaction.end_time = datetime.now()
                db.commit()

            # Update the agent execution log
            agent_execution_log = db.query(AgentExecutionLog).filter(AgentExecutionLog.transaction_id == uuid.UUID(transaction_id)).first()
            if agent_execution_log:
                agent_execution_log.end_time = datetime.now()
                agent_execution_log.output_data = result['result'] if isinstance(result, dict) and 'result' in result else result
                db.commit()
            
            return {
                "session_id": query.session_id,
                "transaction_id": transaction_id,
                "result": result
            }
                
        except Exception as e:
            # Update transaction status to failed
            transaction = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
            if transaction:
                transaction.status = "failed"
                transaction.end_time = datetime.now()
                transaction.final_output = {"error": str(e)}
                db.commit()
            raise
            
    except HTTPException:
        # Re-raise HTTP exceptions (including 400, 404, 500, etc.)
        raise
    except Exception as e:
        logger.error("Error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@agent_router.get("/mcp-tools")
async def get_mcp_tools():
    """
    Fetch available tools from MCP server with metadata.
    Returns all available tools (no permission filtering).
    """
    import httpx
    from utils.mcp_tool_fetcher import fetch_tools_with_metadata
    import os
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    
    try:
        tools_with_metadata = await fetch_tools_with_metadata(mcp_server_url)
        
        if not tools_with_metadata:
            logger.warning("No tools found from MCP server")
            return JSONResponse(content=[])
        
        formatted_tools = [
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "risk_level": tool.get("risk_level", ""),
                "requires_auth": tool.get("requires_auth", False),
                "impact_area": tool.get("impact_area", ""),
                "risk_description": tool.get("risk_description", ""),
                "accepts_datasource_types": tool.get("accepts_datasource_types"),
            }
            for tool in tools_with_metadata
        ]
        
        logger.info(f"Returning {len(formatted_tools)} tools")
        
        return JSONResponse(content=formatted_tools)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching MCP tools: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MCP tools from server: {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MCP tools: {str(e)}"
        )

@agent_router.get("/agents/{agent_id}/memory")
async def get_agent_memory_endpoint(
    agent_id: str,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    try:
        data = get_agent_sessions_and_interactions(db, agent_id, session_id)
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@agent_router.post("/create-session")
async def create_session_endpoint(db: Session = Depends(get_db)):
    """Create a new chat/session."""
    from db.crud.session import create_session
    session_id = uuid.uuid4()
    create_session(db, {
        "id": session_id,
        "start_time": datetime.now(),
        "status": "active"
    })
    return {"session_id": session_id}