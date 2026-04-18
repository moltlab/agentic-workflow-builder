"""
LangChain-specific MCP tools and utilities.
Uses native LangChain MCP adapters for proper LangGraph integration.
"""

import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from db.config import SessionLocal
from db.crud import mcp_server as mcp_server_crud
from utils.logging_utils import get_logger
from utils.mcp_tool_fetcher import fetch_tools_with_metadata
from utils.mcp_url import sse_url
from utils.tool_permissions import can_user_use_tool_by_risk_level
from agents.shared.datasource_scope import apply_datasource_config_to_tools

logger = get_logger('mcp_tools_langchain')

# Default MCP server URL
DEFAULT_MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:8001')


class LangChainMCPToolsManager:
    """Manager class for handling MCP tools specifically for LangChain/LangGraph."""
    
    def __init__(
        self, 
        mcp_config: Dict[str, Any], 
        default_server_url: str = DEFAULT_MCP_SERVER_URL, 
        agent_config: Dict[str, Any] = None,
        user_permissions: Optional[List[str]] = None
    ):
        """
        Initialize LangChain MCP tools manager.
        
        :param mcp_config: MCP configuration from agent config
        :param default_server_url: Default MCP server URL
        :param agent_config: Full agent configuration for tool customization
        :param user_permissions: Optional list of user permission strings for tool filtering
        """
        self.mcp_config = mcp_config
        self.default_server_url = default_server_url
        self.agent_config = agent_config or {}
        self.user_permissions = user_permissions
        self.mcp_client = None
        self._mcp_clients: List[MultiServerMCPClient] = []
        self.langchain_tools = []
        self._tool_metadata_cache = {}  # Cache for tool metadata from /v1/tools
    
    async def initialize_langchain_mcp_tools(self) -> List[BaseTool]:
        """
        Initialize MCP tools using native LangChain MCP adapters.
        Filters tools based on user permissions if provided.
        
        :return: List of LangChain-compatible MCP tools
        """
        if not self.mcp_config.get('enabled', False):
            logger.info("MCP is not enabled for this agent.")
            return []
        
        mcp_server_url = self.mcp_config.get('server_url', self.default_server_url)
        if mcp_server_url == "":
            mcp_server_url = self.default_server_url

        logger.info("LangChain MCP enabled; resolving server URL(s)")

        self._tool_metadata_cache = {}
        groups: Dict[str, Set[str]] = defaultdict(set)
        db = SessionLocal()
        try:
            configured = self.mcp_config.get("tools") or []
            if configured:
                for tcfg in configured:
                    nm = (tcfg.get("name") or "").strip()
                    if not nm:
                        continue
                    resolved = mcp_server_crud.resolve_base_url_for_tool(
                        tcfg, self.mcp_config, db, mcp_server_url
                    )
                    groups[resolved].add(nm)
            else:
                groups[mcp_server_url] = set()

            for base_url in groups:
                try:
                    tools_with_metadata = await fetch_tools_with_metadata(base_url)
                    for t in tools_with_metadata:
                        self._tool_metadata_cache[t.get("name", "")] = t
                except Exception as e:
                    logger.warning("Metadata fetch failed for %s: %s", base_url, e)
        finally:
            db.close()

        try:
            all_filtered: List[BaseTool] = []
            self._mcp_clients.clear()
            for base_url, want_names in groups.items():
                mcp_client_config = {
                    "mcp_server": {
                        "transport": "sse",
                        "url": sse_url(base_url),
                    }
                }
                client = MultiServerMCPClient(mcp_client_config)
                self._mcp_clients.append(client)
                available_tools = await client.get_tools()
                if not available_tools:
                    logger.warning("No tools from LangChain MCP at %s", base_url)
                    continue
                if want_names:
                    filtered_tools = [t for t in available_tools if t.name in want_names]
                else:
                    filtered_tools = list(available_tools)

                if self.user_permissions is not None and self._tool_metadata_cache:
                    permission_filtered_tools = []
                    for tool in filtered_tools:
                        tool_metadata = self._tool_metadata_cache.get(tool.name, {})
                        risk_level = tool_metadata.get("risk_level", "")
                        if can_user_use_tool_by_risk_level(self.user_permissions, risk_level):
                            permission_filtered_tools.append(tool)
                        else:
                            logger.info(
                                "Filtered out tool '%s' (risk_level: %s) — user lacks permission",
                                tool.name,
                                risk_level,
                            )
                    filtered_tools = permission_filtered_tools

                all_filtered.extend(filtered_tools)

            self.mcp_client = self._mcp_clients[0] if self._mcp_clients else None

            if all_filtered and "mcp" in self.agent_config and "tools" in self.agent_config["mcp"]:
                apply_datasource_config_to_tools(
                    all_filtered,
                    self.agent_config["mcp"]["tools"],
                    self._tool_metadata_cache,
                )

            self.langchain_tools = all_filtered
            logger.info("Registered %s LangChain MCP tools", len(all_filtered))

        except Exception as e:
            logger.error("Failed to initialize LangChain MCP client(s): %s", e)

        return self.langchain_tools
    
    def get_tools_for_binding(self) -> List[Dict[str, Any]]:
        """
        Get tools in the format expected by LangChain's bind_tools().
        
        :return: List of tool definitions for LangChain binding
        """
        tool_definitions = []
        
        for tool in self.langchain_tools:
            try:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_def = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    
                    # Extract schema from LangChain tool
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        try:
                            # Handle different schema formats
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                schema = tool.args_schema.model_json_schema()
                            elif hasattr(tool.args_schema, 'schema'):
                                schema = tool.args_schema.schema()
                            elif isinstance(tool.args_schema, dict):
                                schema = tool.args_schema
                            else:
                                schema = None
                                
                            if schema and 'properties' in schema:
                                tool_def["parameters"]["properties"] = schema['properties']
                            if schema and 'required' in schema:
                                tool_def["parameters"]["required"] = schema['required']
                        except Exception as e:
                            logger.warning(f"Could not extract schema from LangChain tool {tool.name}: {e}")
                    
                    # Ensure basic parameters for web_search_tool
                    if tool.name == 'web_search_tool' and not tool_def["parameters"]["properties"]:
                        tool_def["parameters"]["properties"] = {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer", 
                                "description": "Maximum number of results",
                                "default": 5
                            }
                        }
                        tool_def["parameters"]["required"] = ["query"]
                    
                    tool_definitions.append(tool_def)
                    
            except Exception as e:
                logger.warning(f"Could not process LangChain tool {getattr(tool, 'name', 'unknown')}: {e}")
        
        return tool_definitions
    
    def get_tools_for_execution(self) -> List[BaseTool]:
        """
        Get tools for actual execution in LangGraph ToolNode.
        
        :return: List of LangChain BaseTool instances
        """
        return self.langchain_tools
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute a specific MCP tool by name.
        
        :param tool_name: Name of the tool to execute
        :param tool_args: Arguments for the tool
        :return: Tool execution result
        """
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        # Find the tool
        target_tool = None
        for tool in self.langchain_tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                target_tool = tool
                break
        
        if not target_tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            # Execute the LangChain tool
            if hasattr(target_tool, 'invoke'):
                result = await target_tool.invoke(tool_args)
            elif hasattr(target_tool, '_run'):
                result = target_tool._run(**tool_args)
            elif hasattr(target_tool, 'run'):
                result = target_tool.run(**tool_args)
            else:
                raise ValueError(f"Tool {tool_name} has no known execution method")
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing LangChain MCP tool {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup LangChain MCP connections and resources."""
        for client in self._mcp_clients:
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "disconnect"):
                    await client.disconnect()
            except Exception as e:
                logger.error("Error cleaning up LangChain MCP client: %s", e)
        self._mcp_clients.clear()
        self.mcp_client = None
        logger.info("LangChain MCP client(s) cleaned up")


class LangChainMCPToolWrapper(BaseTool):
    """
    Wrapper to make MCP tools compatible with LangChain's tool interface.
    """
    
    def __init__(self, mcp_tool, mcp_manager: LangChainMCPToolsManager):
        super().__init__()
        self.mcp_tool = mcp_tool
        self.mcp_manager = mcp_manager
        self.name = getattr(mcp_tool, 'name', 'unknown_tool')
        self.description = getattr(mcp_tool, 'description', 'MCP tool')
    
    async def _arun(self, **kwargs) -> str:
        """Async execution of the MCP tool."""
        return await self.mcp_manager.execute_tool(self.name, kwargs)
    
    def _run(self, **kwargs) -> str:
        """Sync execution of the MCP tool."""
        import asyncio
        return asyncio.run(self._arun(**kwargs))


def create_langchain_mcp_tools(mcp_manager: LangChainMCPToolsManager) -> List[BaseTool]:
    """
    Factory function to create LangChain-wrapped MCP tools.
    
    :param mcp_manager: Initialized LangChain MCP manager
    :return: List of LangChain-compatible MCP tools
    """
    wrapped_tools = []
    
    for tool in mcp_manager.get_tools_for_execution():
        try:
            wrapped_tool = LangChainMCPToolWrapper(tool, mcp_manager)
            wrapped_tools.append(wrapped_tool)
        except Exception as e:
            logger.warning(f"Could not wrap MCP tool {getattr(tool, 'name', 'unknown')}: {e}")
    
    return wrapped_tools
