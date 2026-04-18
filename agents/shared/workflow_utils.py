"""
Workflow-specific utility functions for hierarchical workflows.
Handles message conversion, node mapping, and other workflow operations.
"""

from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from utils.logging_utils import get_logger

logger = get_logger('workflow_utils')


def build_input_messages(
    db_chat_history: List[Dict[str, Any]],
    query: str,
    storage: Any = None,
) -> List[BaseMessage]:
    """
    Convert database chat history and current query to LangChain messages.
    When storage is provided and an interaction has user_attachments, resolves
    signed URLs and builds multi-modal content for that user turn.

    Args:
        db_chat_history: List of interaction dicts with 'user', 'assistant', optional 'user_attachments'.
        query: Current user query.
        storage: Optional storage instance with get_signed_url() for resolving history attachments.

    Returns:
        List of LangChain BaseMessage objects (HumanMessage and AIMessage).
    """
    from agents.shared.message_builder import build_content, media_items_from_attachment_metadata

    initial_messages = []
    for interaction in db_chat_history:
        user_content = interaction.get("user")
        assistant_content = interaction.get("assistant")
        user_attachments = interaction.get("user_attachments") or []

        if user_content:
            if user_attachments and storage:
                history_media = media_items_from_attachment_metadata(storage, user_attachments)
                content = build_content(user_content, history_media)
            else:
                content = user_content
            initial_messages.append(HumanMessage(content=content))

        if assistant_content is not None:
            initial_messages.append(AIMessage(content=assistant_content))

    initial_messages.append(HumanMessage(content=query))
    return initial_messages


def build_node_agent_map(workflow_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a mapping from node ID to agent ID from workflow configuration.
    
    Args:
        workflow_config: Workflow configuration dictionary with 'nodes' list
        
    Returns:
        Dictionary mapping node_id -> agent_id
    """
    return {
        node.get("id"): node.get("agent")
        for node in workflow_config.get("nodes", [])
        if isinstance(node, dict) and node.get("id") and node.get("agent")
    }

