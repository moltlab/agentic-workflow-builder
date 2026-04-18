from sqlalchemy.orm import Session
from db.models import MemoryStore

def create_memory_store(db: Session, data: dict):
    record = MemoryStore(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_memory_store_by_id(db: Session, id):
    return db.query(MemoryStore).filter(MemoryStore.id == id).first()

def list_memory_stores(db: Session):
    return db.query(MemoryStore).all()