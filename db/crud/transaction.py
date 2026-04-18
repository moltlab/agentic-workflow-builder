from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from db.models import Transaction, AgentExecutionLog, Agent, ToolExecutionLog, Tool
from typing import Optional, List, Dict, Any
from api.schemas.auth import CurrentUser
import uuid as uuid_module


def create_transaction(db: Session, data: dict, current_user: Optional[CurrentUser] = None):
    """
    Create a new transaction with user/entity context.
    
    Args:
        db: Database session
        data: Dictionary with transaction data
        current_user: Optional CurrentUser for auth context
        
    Security:
        Never trusts client-provided entity_id. Always derives from auth context.
    """
    from utils.user_context import enrich_data_with_user_context
    
    # Enrich with user/entity context (removes client-provided values)
    data = enrich_data_with_user_context(data, current_user=current_user)
    
    record = Transaction(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_transaction_by_id(db: Session, id):
    return db.query(Transaction).filter(Transaction.id == id).first()

def list_transactions(db: Session):
    return db.query(Transaction).all()


def get_recent_transactions(
    db: Session,
    limit: int = 10,
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return recent transactions from PostgreSQL for history API.
    Optionally filter by session_id and/or agent name (via AgentExecutionLog).
    """
    q = db.query(Transaction)
    if session_id:
        try:
            sid = uuid_module.UUID(session_id)
            q = q.filter(Transaction.session_id == sid)
        except (ValueError, TypeError):
            return []
    if agent_name:
        subq = (
            db.query(AgentExecutionLog.transaction_id)
            .join(Agent, AgentExecutionLog.agent_id == Agent.id)
            .filter(Agent.name == agent_name)
            .distinct()
            .subquery()
        )
        q = q.filter(Transaction.id.in_(subq))
    rows = q.order_by(desc(Transaction.start_time)).limit(limit).all()
    if not rows:
        return []
    tx_ids = [t.id for t in rows]
    agent_map = {}
    agent_logs = (
        db.query(AgentExecutionLog.transaction_id, Agent.name)
        .join(Agent, AgentExecutionLog.agent_id == Agent.id)
        .filter(AgentExecutionLog.transaction_id.in_(tx_ids))
        .all()
    )
    for tid, name in agent_logs:
        if tid not in agent_map:
            agent_map[tid] = name
    result = []
    for t in rows:
        input_val = t.task_description
        if t.input_data and isinstance(t.input_data, dict) and "query" in t.input_data:
            input_val = t.input_data["query"] or input_val
        out = t.final_output
        if isinstance(out, dict) and "result" in out:
            out = out["result"]
        response_str = str(out) if out is not None else None
        result.append({
            "transaction_id": str(t.id),
            "agent_name": agent_map.get(t.id, "Not specified"),
            "timestamp": t.start_time.isoformat() if t.start_time else None,
            "status": t.status or "",
            "input": input_val or "",
            "duration_ms": t.total_duration_ms,
            "response": response_str,
        })
    return result


def get_session_stats(db: Session, session_id: str) -> Dict[str, Any]:
    """
    Return session statistics from PostgreSQL: transaction count, avg duration, agent and tool usage.
    """
    try:
        sid = uuid_module.UUID(session_id)
    except (ValueError, TypeError):
        return {
            "transaction_count": 0,
            "average_duration_ms": 0,
            "agent_usage": {},
            "tool_usage": {},
        }
    tx_count = db.query(func.count(Transaction.id)).filter(Transaction.session_id == sid).scalar() or 0
    avg_duration = (
        db.query(func.avg(Transaction.total_duration_ms))
        .filter(Transaction.session_id == sid, Transaction.total_duration_ms.isnot(None))
        .scalar()
    )
    avg_duration = float(avg_duration) if avg_duration is not None else 0.0
    tx_subq = db.query(Transaction.id).filter(Transaction.session_id == sid).subquery()
    agent_rows = (
        db.query(Agent.name, func.count(AgentExecutionLog.id))
        .join(AgentExecutionLog, AgentExecutionLog.agent_id == Agent.id)
        .filter(AgentExecutionLog.transaction_id.in_(tx_subq))
        .group_by(Agent.name)
        .all()
    )
    agent_usage = {name: count for name, count in agent_rows}
    tool_rows = (
        db.query(Tool.name, func.count(ToolExecutionLog.id))
        .join(ToolExecutionLog, ToolExecutionLog.tool_id == Tool.id)
        .filter(ToolExecutionLog.transaction_id.in_(tx_subq))
        .group_by(Tool.name)
        .all()
    )
    tool_usage = {name: count for name, count in tool_rows}
    return {
        "transaction_count": tx_count,
        "average_duration_ms": avg_duration,
        "agent_usage": agent_usage,
        "tool_usage": tool_usage,
    }