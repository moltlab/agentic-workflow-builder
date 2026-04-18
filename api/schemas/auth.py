"""
Authentication Schemas

Pydantic models for authentication requests and responses.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List


class LoginRequest(BaseModel):
    """Request model for login"""
    identifier: str  # Email or phone number
    password: str
    entity: Optional[str] = None  # Optional entity domain


class LoginResponse(BaseModel):
    """Response model for successful login"""
    success: bool
    token: str
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    primary_role: Optional[str] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    signup_status: Optional[str] = None


class LogoutResponse(BaseModel):
    """Response model for logout"""
    success: bool
    message: str


class UserContext(BaseModel):
    """User context extracted from token"""
    user_id: str
    entity_id: Optional[str] = None
    roles: List[str] = []
    primary_role: Optional[str] = None
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class CurrentUser(BaseModel):
    """Current authenticated user (used in route dependencies)"""
    user_id: str
    entity_id: Optional[str] = None
    roles: List[str] = []
    primary_role: Optional[str] = None
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    token: str  # Include token for permission checks
    
    @classmethod
    def from_user_context(cls, user_context: UserContext, token: str) -> "CurrentUser":
        """Create CurrentUser from UserContext and token"""
        return cls(
            user_id=user_context.user_id,
            entity_id=user_context.entity_id,
            roles=user_context.roles,
            primary_role=user_context.primary_role,
            email=user_context.email,
            first_name=user_context.first_name,
            last_name=user_context.last_name,
            token=token
        )
