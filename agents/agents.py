from utils.logging_utils import get_logger
from typing import Dict, Any
from sqlalchemy.orm import Session
from db.crud.agent import get_agent_by_id
from .base_agent import BaseAgent

logger = get_logger('agents')

class AgentFactory:
    """Factory class for creating agents from database definitions."""
    
    @classmethod
    def create_agent(cls, agent_data: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent instance based on the database definition.
        
        :param agent_data: Dictionary containing agent configuration from database
                         Expected format:
                         {
                             'name': 'Agent Name',
                             'description': 'Agent Description',
                             'config': {
                                 'llms': {
                                     'openai': {
                                         'api_key': 'your-api-key',
                                         'model': 'gpt-4',
                                         'temperature': 0.7
                                     }
                                 },
                                 'mcp': {
                                     'enabled': True/False,
                                     'server_url': 'http://localhost:8000'
                                 },
                                 'prompt_template': 'Your system prompt here'
                             }
                         }
        :return: An instance of BaseAgent
        """
        required_fields = ['name', 'description', 'config']
        for field in required_fields:
            if field not in agent_data:
                raise ValueError(f"Missing required field: {field}")
        
        return BaseAgent(
            name=agent_data['name'],
            description=agent_data['description'],
            config=agent_data['config']
        )

async def get_and_run_agent(db: Session, agent_id: str, query: str) -> str:
    """
    Get an agent from the database, create an instance, and run it.
    
    :param db: Database session
    :param agent_id: ID of the agent in the database
    :param query: Query to run with the agent
    :return: Agent's response
    """
    # Get agent from database
    agent_db = get_agent_by_id(db, agent_id)
    if not agent_db:
        raise ValueError(f"Agent with ID {agent_id} not found")
    
    # Convert to dictionary format expected by factory
    agent_data = {
        'name': agent_db.name,
        'description': agent_db.description,
        'config': agent_db.config
    }
    
    # Create agent
    agent = AgentFactory.create_agent(agent_data)
    
    try:
        # Initialize agent (and MCP tools if enabled)
        await agent.initialize()
        print("is_mcp_enabled", agent.is_mcp_enabled)
        # Check if this is a calculation request and MCP is enabled
        if agent.is_mcp_enabled and ("calculate" in query.lower() or "add" in query.lower()):
            try:
                # Extract numbers from query
                import re
                numbers = re.findall(r'\d+', query)
                if len(numbers) >= 2:
                    # Get available tools
                    tools = await agent.get_available_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    print("Available tools:", tool_names)
                    if "add_tool" in tool_names:
                        # Use the calculator tool
                        result = await agent.use_tool("add_tool", arguments={"a": int(numbers[0]), "b": int(numbers[1])})
                        return f"Calculation result: {result}"
                    else:
                        logger.warning("Calculator tool not available on MCP server")
            except Exception as e:
                logger.error(f"Error using MCP calculator tool: {str(e)}")
        
        # Execute the agent's main task
        print("Executing agent with agent id", agent_id)
        response = agent.execute(query)
        return response
        
    finally:
        # Cleanup resources
        await agent.cleanup()