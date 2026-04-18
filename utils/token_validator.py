"""
Token Validator

This module provides JWT token validation and user context extraction.
"""

import os
import jwt
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logging_utils import get_logger

logger = get_logger('token_validator')


class TokenValidator:
    """Validates JWT tokens and extracts user information"""
    
    def __init__(self, secret: Optional[str] = None, algorithm: str = "HS256"):
        """
        Initialize token validator.
        
        Args:
            secret: JWT secret key. If not provided, uses ONBOARDO_JWT_SECRET env var.
                   If still not provided, token validation will be skipped (not recommended).
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret = secret or os.getenv('ONBOARDO_JWT_SECRET')
        self.algorithm = algorithm
        
        # If no secret is provided, we'll still decode the token to extract claims
        # but won't verify the signature. This is useful if OnboardO tokens are
        # self-contained and we trust the issuer.
        if not self.secret:
            logger.warning("No JWT secret provided. Token signature will not be verified.")
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and extract claims.
        
        Args:
            token: JWT token string
        
        Returns:
            Dict containing token claims (user_id, entity_id, roles, etc.)
        
        Raises:
            ValueError: If token is invalid or expired
            jwt.ExpiredSignatureError: If token has expired
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            # Decode token
            # If secret is provided, verify signature
            # If not, just decode (for self-contained tokens from trusted issuer)
            if self.secret:
                payload = jwt.decode(
                    token,
                    self.secret,
                    algorithms=[self.algorithm],
                    options={"verify_signature": True}
                )
            else:
                # Decode without verification (for self-contained tokens)
                payload = jwt.decode(
                    token,
                    options={"verify_signature": False}
                )
            
            # Check expiration
            exp = payload.get('exp')
            if exp:
                exp_timestamp = datetime.fromtimestamp(exp)
                if datetime.now() > exp_timestamp:
                    raise jwt.ExpiredSignatureError("Token has expired")
            
            # Log at DEBUG level to reduce noise (token validation happens on every request)
            logger.debug(f"Token validated successfully for user: {payload.get('user_id')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise ValueError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise ValueError(f"Token validation failed: {str(e)}")
    
    def extract_user_context(self, token: str) -> Dict[str, Any]:
        """
        Extract user context from token.
        
        Args:
            token: JWT token string
        
        Returns:
            Dict containing user context:
            - user_id: User ID
            - entity_id: Entity ID
            - roles: List of roles
            - email: User email
            - first_name: User first name
            - last_name: User last name
        """
        payload = self.validate_token(token)
        
        return {
            "user_id": payload.get("user_id"),
            "entity_id": payload.get("entity_id"),
            "roles": payload.get("roles", []),
            "primary_role": payload.get("primary_role"),
            "email": payload.get("email"),
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name"),
            "exp": payload.get("exp"),
        }


# Singleton instance
_validator_instance: Optional[TokenValidator] = None


def get_token_validator() -> TokenValidator:
    """
    Get or create the singleton token validator instance.
    
    Returns:
        TokenValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TokenValidator()
    return _validator_instance
