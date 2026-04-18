from sqlalchemy.orm import Session
from db.models import MemoryTransactionLog

def create_memory_transaction_log(db: Session, data: dict):
    record = MemoryTransactionLog(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_memory_transaction_log_by_id(db: Session, id):
    return db.query(MemoryTransactionLog).filter(MemoryTransactionLog.id == id).first()

def list_memory_transaction_logs(db: Session):
    return db.query(MemoryTransactionLog).all()