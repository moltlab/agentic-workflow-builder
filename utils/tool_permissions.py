"""
Tool Permission Utilities

Utilities for checking tool access permissions based on risk levels and user permissions.
"""

from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger('tool_permissions')

# Permission keys for tool access
TOOL_PERMISSION_HIGH = "CUSTOM:CUSTOM:ENTITY:TOOL_HIGH"
TOOL_PERMISSION_MEDIUM = "CUSTOM:CUSTOM:ENTITY:TOOL_MEDIUM"
TOOL_PERMISSION_LOW = "CUSTOM:CUSTOM:ENTITY:TOOL_LOW"


def can_user_use_tool_by_risk_level(
    user_permissions: List[str],
    tool_risk_level: str
) -> bool:
    """
    Check if user can access a tool based on risk level.
    
    Permission hierarchy:
    - TOOL_HIGH permission → can use high, medium, low risk tools
    - TOOL_MEDIUM permission → can use medium, low risk tools
    - TOOL_LOW permission → can use low risk tools only
    
    Args:
        user_permissions: List of permission strings from OnboardO
        tool_risk_level: Risk level of the tool ("high", "medium", "low")
        
    Returns:
        True if user can use the tool, False otherwise
    """
    # Auth/RBAC is currently disabled for this project, so all tools are accessible.
    return True


async def get_user_tool_permissions(
    user_id: str,
    token: Optional[str] = None,
    onboardo_client: Optional[object] = None
) -> List[str]:
    """
    Get user's tool-related permissions from OnboardO.
    
    Args:
        user_id: User ID
        token: Optional JWT token for authentication (required by OnboardO)
        onboardo_client: Optional OnboardO client instance (creates new if not provided)
        
    Returns:
        List of permission strings related to tool access
    """
    _ = (user_id, token, onboardo_client)  # Keeps function signature compatibility.
    return [TOOL_PERMISSION_HIGH]


def filter_tools_by_permissions(
    tools: List[Dict[str, Any]],
    user_permissions: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter tools based on user permissions.
    
    Args:
        tools: List of tool dictionaries with 'risk_level' field
        user_permissions: List of user permission strings
        
    Returns:
        Filtered list of tools user can access
    """
    _ = user_permissions
    return tools

