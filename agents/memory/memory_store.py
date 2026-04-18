from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict, Any, Optional
import json

class ChatMemory:
    """Manages conversation history for agents using LangChain's ConversationBufferMemory."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the chat memory store.
        
        Args:
            max_history (int): Maximum number of messages to keep in history
        """
        self.memory = ConversationBufferMemory(return_messages=True)
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): Role of the message sender ('user' or 'assistant')
            content (str): Content of the message
            metadata (Dict[str, Any], optional): Additional metadata for the message
        """
        # Add to LangChain memory
        if role == 'user':
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)
        
        # Add to our history
        message = {
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        self._history.append(message)
        
        # Trim history if it exceeds max_history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self._history
    
    def get_formatted_history(self, max_messages: Optional[int] = None) -> str:
        """
        Get a formatted string of recent messages for use in prompts.
        
        Args:
            max_messages (int, optional): Maximum number of recent messages to include
        """
        history = self._history
        if max_messages:
            history = history[-max_messages:]
        
        formatted = []
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
        self._history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory store to a dictionary for serialization."""
        return {
            'max_history': self.max_history,
            'history': self._history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMemory':
        """Create a ChatMemory instance from a dictionary."""
        memory = cls(max_history=data.get('max_history', 10))
        for msg in data.get('history', []):
            memory.add_message(
                role=msg['role'],
                content=msg['content'],
                metadata=msg.get('metadata')
            )
        return memory 