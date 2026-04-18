"""
Authentication Middleware

FastAPI dependencies for authentication and user context extraction.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from utils.token_validator import get_token_validator, TokenValidator
from api.schemas.auth import CurrentUser, UserContext
from utils.logging_utils import get_logger

logger = get_logger('auth_middleware')

# HTTP Bearer token security scheme
security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> None:
    """
    Verify token as a router-level dependency.
    This ensures all routes in the router require authentication.
    The CurrentUser is stored in request.state for access in routes.
    
    Note: Router-level dependencies don't return values that routes can use directly.
    Routes that need user info should use get_current_user() dependency.
    """
    token = credentials.credentials
    
    if not token:
        logger.warning("No token provided in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        validator = get_token_validator()
        user_context_dict = validator.extract_user_context(token)
        user_context = UserContext(**user_context_dict)
        current_user = CurrentUser.from_user_context(user_context, token)
        
        # Store in request.state for easy access in routes
        if request:
            request.state.current_user = current_user
        
        logger.debug(f"Token verified for user: {current_user.user_id}")
        # Router-level dependencies don't need to return a value
        
    except ValueError as e:
        logger.warning(f"Token validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> CurrentUser:
    """
    FastAPI dependency to extract and validate current user from JWT token.
    
    This dependency:
    1. Extracts the Bearer token from the Authorization header
    2. Validates the token
    3. Extracts user context from the token
    4. Returns CurrentUser object
    
    Can be used in two ways:
    1. Router-level: APIRouter(dependencies=[Depends(verify_token)]) - ensures auth
    2. Route-level: current_user: CurrentUser = Depends(get_current_user) - gets user info
    
    Usage:
        # Router-level (ensures all routes are authenticated)
        router = APIRouter(dependencies=[Depends(verify_token)])
        
        # Route-level (when you need user info)
        @router.get("/protected")
        async def protected_route(current_user: CurrentUser = Depends(get_current_user)):
            return {"user_id": current_user.user_id}
        
        # Or access from request.state (if verify_token was used as router dependency)
        @router.get("/protected")
        async def protected_route(request: Request):
            current_user = request.state.current_user
            return {"user_id": current_user.user_id}
    
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    token = credentials.credentials
    
    if not token:
        logger.warning("No token provided in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        validator = get_token_validator()
        user_context_dict = validator.extract_user_context(token)
        user_context = UserContext(**user_context_dict)
        current_user = CurrentUser.from_user_context(user_context, token)
        
        logger.debug(f"Authenticated user: {current_user.user_id}")
        return current_user
        
    except ValueError as e:
        logger.warning(f"Token validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[CurrentUser]:
    """
    FastAPI dependency to optionally extract user from token.
    
    Unlike get_current_user, this doesn't raise an error if no token is provided.
    Returns None if no token or invalid token.
    
    Usage:
        @app.get("/public-or-protected")
        async def route(user: Optional[CurrentUser] = Depends(get_optional_user)):
            if user:
                return {"user_id": user.user_id}
            return {"message": "Anonymous access"}
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    if not token:
        return None
    
    try:
        validator = get_token_validator()
        user_context_dict = validator.extract_user_context(token)
        user_context = UserContext(**user_context_dict)
        return CurrentUser.from_user_context(user_context, token)
    except Exception:
        # If token is invalid, treat as no user
        return None


async def require_auth_for_html(request: Request):
    """
    Dependency for HTML routes that requires authentication.
    Checks for token in Authorization header or cookie.
    Redirects to /login if token is missing or invalid.
    
    Usage:
        @app.get("/protected-page", response_class=HTMLResponse)
        async def protected_page(request: Request, _ = Depends(require_auth_for_html)):
            return templates.TemplateResponse("page.html", {"request": request})
    """
    from fastapi.responses import RedirectResponse
    
    # Try to get token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    token = None
    
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        # Try to get from cookie
        token = request.cookies.get("authToken")
    
    # If no token found, redirect to login
    if not token:
        logger.warning("No token found for HTML route, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
    
    # Validate token
    try:
        validator = get_token_validator()
        user_context_dict = validator.extract_user_context(token)
        user_context = UserContext(**user_context_dict)
        current_user = CurrentUser.from_user_context(user_context, token)
        
        # Store in request.state for use in route
        request.state.current_user = current_user
        logger.debug(f"HTML route authenticated for user: {current_user.user_id}")
        return None  # Return None to allow request to proceed
        
    except (ValueError, Exception) as e:
        logger.warning(f"Invalid token for HTML route: {str(e)}, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
