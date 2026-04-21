"""
Authentication Routes

API endpoints for login, logout, and user information.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Response
from api.schemas.auth import LoginRequest, LoginResponse, LogoutResponse, CurrentUser
from api.middleware.auth_middleware import get_current_user
from utils.logging_utils import get_logger

logger = get_logger('auth_routes')

auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])


@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
        logger.info(f"User logged in:")
        return 
        



@auth_router.post("/logout", response_model=LogoutResponse)
async def logout(current_user: CurrentUser = Depends(get_current_user)):
        logger.info(f"User logged out:")
        return


@auth_router.get("/me")
async def get_current_user_info(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get current user information from token.
    
    Returns the user information extracted from the JWT token.
    """
    return {
        "user_id": current_user.user_id,
        "entity_id": current_user.entity_id,
        "roles": current_user.roles,
        "primary_role": current_user.primary_role,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name
    }
