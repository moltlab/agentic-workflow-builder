"""
Authentication Routes

API endpoints for login, logout, and user information.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Response
from api.schemas.auth import LoginRequest, LoginResponse, LogoutResponse, CurrentUser
from api.middleware.auth_middleware import get_current_user
from utils.onboardo_client import OnboardOClient, get_onboardo_client
from utils.logging_utils import get_logger

logger = get_logger('auth_routes')

auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])


@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login endpoint that proxies to OnboardO.
    
    Accepts email/phone and password, returns JWT token and user information.
    Also sets a cookie with the token for server-side validation.
    """
    from fastapi import Response as FastAPIResponse
    
    try:
        client = get_onboardo_client()
        
        # Call OnboardO login endpoint
        result = await client.login(
            identifier=request.identifier,
            password=request.password,
            entity=request.entity
        )
        
        token = result.get("token")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No token received from authentication service"
            )
        
        # Map OnboardO response to our response model
        login_response = LoginResponse(
            success=result.get("success", True),
            token=token,
            user_id=result.get("user_id"),
            first_name=result.get("first_name"),
            last_name=result.get("last_name"),
            email=result.get("email"),
            primary_role=result.get("primary_role"),
            entity_id=result.get("entity_id"),
            entity_name=result.get("entity_name"),
            signup_status=result.get("signup_status")
        )
        
        # Create response with cookie
        # Note: We need to return a Response object to set cookies
        from fastapi.responses import JSONResponse
        response = JSONResponse(content=login_response.dict())
        
        # Set cookie with the token for server-side validation
        # Cookie expires in 7 days (adjust based on token expiry)
        # httponly=False allows JavaScript access (needed for API calls from client)
        # SameSite='Lax' prevents CSRF attacks while allowing normal navigation
        response.set_cookie(
            key="authToken",
            value=token,
            max_age=7 * 24 * 60 * 60,  # 7 days in seconds
            httponly=False,  # Allow JavaScript access for API calls
            samesite="lax",  # CSRF protection
            secure=False,  # Set to True in production with HTTPS
            path="/"  # Available for all paths
        )
        
        logger.info(f"User logged in: {login_response.user_id} ({login_response.email})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}"
        )


@auth_router.post("/logout", response_model=LogoutResponse)
async def logout(current_user: CurrentUser = Depends(get_current_user)):
    """
    Logout endpoint.
    
    Attempts to revoke the token via OnboardO (if endpoint exists).
    Clears the auth cookie.
    Even if OnboardO doesn't have a logout endpoint, the token will expire naturally.
    """
    from fastapi.responses import JSONResponse
    
    try:
        client = get_onboardo_client()
        
        # Try to logout via OnboardO (may not be available)
        try:
            await client.logout(current_user.token)
            logger.info(f"User logged out: {current_user.user_id}")
        except Exception as e:
            # If logout endpoint doesn't exist, that's okay
            logger.info(f"Logout endpoint not available, token will expire naturally: {str(e)}")
        
        # Create response and clear the auth cookie
        logout_response = LogoutResponse(
            success=True,
            message="Logged out successfully"
        )
        response = JSONResponse(content=logout_response.dict())
        response.delete_cookie(
            key="authToken",
            path="/",
            samesite="lax"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        # Still clear cookie and return success since token will expire
        logout_response = LogoutResponse(
            success=True,
            message="Logged out (token will expire naturally)"
        )
        response = JSONResponse(content=logout_response.dict())
        response.delete_cookie(
            key="authToken",
            path="/",
            samesite="lax"
        )
        return response


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
