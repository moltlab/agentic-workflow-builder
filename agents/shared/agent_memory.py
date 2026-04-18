"""
Generic agent memory management functionality.
Can be used across CrewAI, LangGraph, and other agent frameworks.
"""

from typing import Dict
from memory.memory_store import ChatMemory

# Global dictionary to store agent memories
agent_memories: Dict[str, Dict[str, ChatMemory]] = {}


def get_agent_memory(agent_id: str, session_id: str, max_history: int = 10) -> ChatMemory:
    """
    Get or create a memory store for an agent and session.
    Generic function that works across all agent frameworks.
    
    Args:
        agent_id (str): ID of the agent
        session_id (str): ID of the user session
        max_history (int): Maximum number of messages to keep in history
    
    Returns:
        ChatMemory: The agent's memory store for this session
    """
    if agent_id not in agent_memories:
        agent_memories[agent_id] = {}
    
    if session_id not in agent_memories[agent_id]:
        agent_memories[agent_id][session_id] = ChatMemory(max_history=max_history)
    
    return agent_memories[agent_id][session_id]


def get_memory_config(agent_config: Dict) -> Dict[str, int]:
    """
    Extract memory configuration from agent config.
    
    Args:
        agent_config: Agent configuration dictionary
        
    Returns:
        Dictionary with memory configuration values
    """
    memory_config = agent_config.get('memory', {})
    return {
        'max_history': memory_config.get('max_history', 10),
        'context_window': memory_config.get('context_window', 5)
    }


def prepare_chat_history(memory: ChatMemory, context_window: int = 5) -> str:
    """
    Prepare chat history for inclusion in agent prompts.
    
    Args:
        memory: ChatMemory instance
        context_window: Number of recent messages to include
        
    Returns:
        Formatted chat history string
    """
    formatted_history = memory.get_formatted_history(max_messages=context_window)
    return formatted_history or "No previous conversation history."
