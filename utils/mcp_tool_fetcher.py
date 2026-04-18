"""
MCP Tool Fetcher

Utility to fetch tools with metadata from MCP server's /v1/tools endpoint.
"""

import httpx
import os
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger('mcp_tool_fetcher')


async def fetch_tools_with_metadata(
    mcp_server_url: Optional[str] = None,
    timeout: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Fetch tools with metadata from MCP server's /v1/tools endpoint.
    
    Args:
        mcp_server_url: MCP server base URL (defaults to MCP_SERVER_URL env var)
        timeout: Request timeout in seconds
        
    Returns:
        List of tool dictionaries with metadata including:
        - name: Tool name
        - description: Tool description
        - risk_level: Tool risk level (high/medium/low)
        - requires_auth: Whether tool requires authentication
        - impact_area: Impact area (read/write/etc)
        - risk_description: Description of risk
        
    Raises:
        httpx.HTTPStatusError: If HTTP request fails
        Exception: For other errors
    """
    if mcp_server_url is None:
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    
    # Remove trailing slash and /mcp suffix if present
    mcp_server_url = mcp_server_url.rstrip('/')
    if mcp_server_url.endswith('/mcp'):
        mcp_server_url = mcp_server_url[:-4]
    
    # Construct /v1/tools endpoint URL
    tools_url = f"{mcp_server_url}/v1/tools"
    
    logger.info(f"Fetching tools with metadata from {tools_url}")
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(tools_url)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, dict):
                tools = data.get("tools", [])
            elif isinstance(data, list):
                tools = data
            else:
                logger.error(f"Unexpected response format from {tools_url}: {type(data)}")
                return []
            
            logger.info(f"Successfully fetched {len(tools)} tools with metadata")
            return tools
            
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error fetching tools from {tools_url}: "
            f"{e.response.status_code} - {e.response.text}"
        )
        raise
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching tools from {tools_url}")
        raise
    except Exception as e:
        logger.error(f"Error fetching tools from {tools_url}: {str(e)}")
        raise

