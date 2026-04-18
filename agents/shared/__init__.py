"""
Shared agent functionality that can be used across different frameworks.
Includes LLM initialization, logging, memory management, MCP tools, and more.
"""

# Core factories and managers
from .llm_factory import LLMFactory
from .mcp_tools import create_mcp_resource_tools
from .mcp_tools_langchain import LangChainMCPToolsManager, create_langchain_mcp_tools
from .template_manager import TemplateManager
from .database_utils import AgentDatabaseManager
from .workflow_utils import build_input_messages, build_node_agent_map
from .message_builder import build_content as build_multimodal_content, build_crew_task_description

# Memory management
from .agent_memory import get_agent_memory, get_memory_config, prepare_chat_history

# Logging and tracing
from .agent_logging import (
    log_agent_step_with_tracing,
    log_agent_step_original,
    log_agent_step_langgraph,
    create_agent_execution_span,
    log_llm_usage,
    log_agent_interactions,
    serialize_tool_usage,
    serialize_datetime
)

# MCP Tools
from .mcp_tools import ResourceAccessTool, ListFilesTool, SummarizeTextTool, FindKeywordsTool

__all__ = [
    # Factories and managers
    'LLMFactory',
    'LangChainMCPToolsManager',
    'TemplateManager',
    'AgentDatabaseManager',
    
    # Memory management
    'get_agent_memory',
    'get_memory_config',
    'prepare_chat_history',
    
    # Logging and tracing
    'log_agent_step_with_tracing',
    'log_agent_step_original',
    'log_agent_step_langgraph',
    'create_agent_execution_span',  
    'log_llm_usage',
    'log_agent_interactions',
    'serialize_tool_usage',
    'serialize_datetime',
    
    # Workflow utilities
    'build_input_messages',
    'build_node_agent_map',
    'build_multimodal_content',
    'build_crew_task_description',
    
    # MCP Tools
    'create_mcp_resource_tools',
    'create_langchain_mcp_tools',
    'ResourceAccessTool',
    'ListFilesTool',
    'SummarizeTextTool', 
    'FindKeywordsTool'
]
