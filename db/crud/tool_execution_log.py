from sqlalchemy.orm import Session
from db.models import ToolExecutionLog

def create_tool_execution_log(db: Session, data: dict):
    record = ToolExecutionLog(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_tool_execution_log_by_id(db: Session, id):
    return db.query(ToolExecutionLog).filter(ToolExecutionLog.id == id).first()

def list_tool_execution_logs(db: Session):
    return db.query(ToolExecutionLog).all()