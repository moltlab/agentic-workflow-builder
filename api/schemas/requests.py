from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    """Schema for query requests"""
    query: str
    agent: str = ""
    session_id: Optional[str] = None
    framework: Optional[str] = "langgraph"
    media_ids: Optional[List[str]] = None

class DocumentRequest(BaseModel):
    """Schema for document requests"""
    doc_id: Optional[str] = None

class HistoryRequest(BaseModel):
    """Schema for history requests"""
    limit: int = 10
    agent: Optional[str] = None
    session_id: Optional[str] = None

class TransactionResponse(BaseModel):
    """Schema for transaction responses"""
    session_id: str
    transaction_id: str
    result: Any
    duration_ms: int

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    detail: str 