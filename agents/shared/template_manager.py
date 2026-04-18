"""
Generic template management functionality.
Can be used across CrewAI, LangGraph, and other agent frameworks.
"""

from typing import Optional, Dict, Any
try:
    from agents.templates.template_loader import AgentTemplateLoader
except ImportError:
    class AgentTemplateLoader:  # type: ignore[override]
        """Fallback loader when template package is not available."""

        def get_backstory(self, agent_name: str) -> Optional[str]:
            _ = agent_name
            return None

        def get_output_format(self, agent_name: str) -> Optional[str]:
            _ = agent_name
            return None

# Initialize template loader (singleton pattern)
template_loader = AgentTemplateLoader()


class TemplateManager:
    """Manager class for handling agent templates across different frameworks."""
    
    @staticmethod
    def get_agent_backstory(agent_name: str, agent_description: str, 
                           prompt_template: Optional[str] = None,
                           chat_history: str = "") -> str:
        """
        Get agent backstory/system prompt from templates or prompt_template.
        
        Args:
            agent_name: Name of the agent
            agent_description: Description of the agent
            prompt_template: Custom prompt template from database
            chat_history: Formatted chat history
            
        Returns:
            Complete backstory/system prompt for the agent
        """
        # Use prompt template if provided and not empty
        if prompt_template and prompt_template not in ["None", "", None]:
            backstory = prompt_template
        else:
            # Try to load from template file
            backstory = template_loader.get_backstory(agent_name)
            
            if not backstory:
                # Use default backstory if no template exists
                backstory = f"""You are {agent_name}, {agent_description}

Please use the output from the tool to answer the user's query in a proper sentence.
If the tool output is not relevant to the user's query, please use the LLM to answer the query."""
        
        # Add chat history if available
        if chat_history:
            backstory += f"""
                
Recent conversation history:
{chat_history}
"""
        
        return backstory
    
    @staticmethod
    def get_agent_output_format(agent_name: str, output_format: Optional[str] = None) -> str:
        """
        Get expected output format for agent from templates or database.
        
        Args:
            agent_name: Name of the agent
            output_format: Custom output format from database
            
        Returns:
            Expected output format for the agent
        """
        if output_format and output_format not in ["None", "", None]:
            return output_format
        
        # Load from template file
        template_output_format = template_loader.get_output_format(agent_name)
        return template_output_format or "A detailed response to the user's query."
    
    @staticmethod
    def prepare_agent_prompt(agent_name: str, agent_description: str, 
                            query: str, agent_id: str, session_id: str,
                            chat_history: str = "", 
                            prompt_template: Optional[str] = None,
                            tools_info: str = "") -> str:
        """
        Prepare complete agent prompt including backstory, context, and metadata.
        
        Args:
            agent_name: Name of the agent
            agent_description: Description of the agent
            query: User query
            agent_id: Agent ID
            session_id: Session ID
            chat_history: Formatted chat history
            prompt_template: Custom prompt template
            tools_info: Information about available tools
            
        Returns:
            Complete agent prompt
        """
        backstory = TemplateManager.get_agent_backstory(
            agent_name, agent_description, prompt_template, chat_history
        )
        
        # Add metadata
        db_details = f"agent_id: {agent_id}, session_id: {session_id}"
        backstory += f"\n\n{db_details}"
        
        # Add tools information if available
        if tools_info:
            backstory += f"\n\n{tools_info}"
        
        return backstory
