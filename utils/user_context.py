"""
User Context Utilities

Helper functions for extracting and validating user_id and entity_id from authentication context.
"""

from typing import Optional, Tuple
from api.schemas.auth import CurrentUser
from sqlalchemy.orm import Session
from db.models import Transaction
import uuid

# Phase 0 defaults
DEFAULT_USER_ID = "system"
DEFAULT_ENTITY_ID = "default_entity"


def resolve_user_entity(current_user: Optional[CurrentUser] = None) -> Tuple[str, str]:
    """
    Safely extract user_id and entity_id from current_user with Phase 0 defaults.
    
    Args:
        current_user: CurrentUser object from auth middleware, or None
        
    Returns:
        Tuple of (user_id, entity_id) with defaults applied
        
    Security Note:
        Never trusts client-provided entity_id. Always extracts from authenticated token.
    """
    if current_user:
        user_id = getattr(current_user, "user_id", None) or DEFAULT_USER_ID
        entity_id = getattr(current_user, "entity_id", None) or DEFAULT_ENTITY_ID
    else:
        user_id = DEFAULT_USER_ID
        entity_id = DEFAULT_ENTITY_ID
    
    return user_id, entity_id


def validate_and_override_entity_id(
    data: dict, 
    current_user: Optional[CurrentUser] = None
) -> dict:
    """
    Remove client-provided entity_id from data dict and replace with authenticated value.
    
    Security: Never trust client-sent entity_id. Always derive from token.
    
    Args:
        data: Dictionary that may contain entity_id or user_id
        current_user: CurrentUser object from auth middleware
        
    Returns:
        Dictionary with entity_id and user_id overridden from auth context
    """
    user_id, entity_id = resolve_user_entity(current_user)
    
    # Remove client-provided values (security: never trust client)
    data.pop("entity_id", None)
    data.pop("user_id", None)
    data.pop("created_by_user_id", None)
    
    # Set from authenticated context
    data["entity_id"] = entity_id
    data["user_id"] = user_id
    
    return data


def enrich_data_with_user_context(
    data: dict,
    current_user: Optional[CurrentUser] = None,
    include_user_id: bool = True,
    include_entity_id: bool = True,
    user_id_field: str = "user_id",
    entity_id_field: str = "entity_id"
) -> dict:
    """
    Enrich data dictionary with user_id and entity_id from auth context.
    
    Args:
        data: Dictionary to enrich
        current_user: CurrentUser object from auth middleware
        include_user_id: Whether to add user_id
        include_entity_id: Whether to add entity_id
        user_id_field: Field name for user_id (default: "user_id")
        entity_id_field: Field name for entity_id (default: "entity_id")
        
    Returns:
        Dictionary with user_id and entity_id added/overridden
    """
    user_id, entity_id = resolve_user_entity(current_user)
    
    # Remove any client-provided values (security)
    if include_user_id:
        data.pop(user_id_field, None)
        data.pop("created_by_user_id", None)
        data[user_id_field] = user_id
    
    if include_entity_id:
        data.pop(entity_id_field, None)
        data[entity_id_field] = entity_id
    
    return data


def get_user_context_from_transaction(db: Session, transaction_id: uuid.UUID) -> Tuple[str, str]:
    """
    Extract user_id and entity_id from a transaction.
    
    This is used in agent execution code where we have transaction_id but not current_user.
    
    Args:
        db: Database session
        transaction_id: Transaction UUID
        
    Returns:
        Tuple of (user_id, entity_id)
        
    Raises:
        ValueError: If transaction not found
    """
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not transaction:
        raise ValueError(f"Transaction {transaction_id} not found")
    
    user_id = transaction.user_id or DEFAULT_USER_ID
    entity_id = transaction.entity_id or DEFAULT_ENTITY_ID
    
    return user_id, entity_id
