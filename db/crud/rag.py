from sqlalchemy.orm import Session
from db.models import RAGExecutionLog

def create_rag_execution_log(db: Session, data: dict):
    record = RAGExecutionLog(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_rag_execution_log_by_id(db: Session, id):
    return db.query(RAGExecutionLog).filter(RAGExecutionLog.id == id).first()

def list_rag_execution_logs(db: Session):
    return db.query(RAGExecutionLog).all()