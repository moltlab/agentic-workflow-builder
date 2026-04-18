from sqlalchemy.orm import Session
from db.models import PerformanceMetric

def create_performance_metric(db: Session, data: dict):
    record = PerformanceMetric(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_performance_metric_by_id(db: Session, id):
    return db.query(PerformanceMetric).filter(PerformanceMetric.id == id).first()

def list_performance_metrics(db: Session):
    return db.query(PerformanceMetric).all()