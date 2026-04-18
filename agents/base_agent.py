from abc import ABC
import logging
from typing import Dict, Any, List
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

class BaseAgent(ABC):
    """
    Base class for AI agents that defines a common interface.
    All agents must implement the `execute` method.
    """

    def __init__(self, name: str, description: str = None, config: Dict[str, Any] = None, role: str = None):
        """
        Initialize the agent with a name, description, and configuration.

        :param name: Name of the agent
        :param description: Description of the agent's role (optional)
        :param config: Configuration dictionary containing LLM and MCP settings (optional)
        :param role: Role of the agent (optional, for backward compatibility)
        """
        self.name = name
        self.description = description or role or ""  # Use role as description if description is not provided
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
        # logging.basicConfig(level=logging.INFO)
        
        # Initialize LLM client if config is provided
        if self.config:
            self._initialize_llm_client()
            
            # Initialize MCP adapter if enabled
            self.is_mcp_enabled = self.config.get('mcp', {}).get('enabled', False)
            if self.is_mcp_enabled:
                server_url = self.config.get('mcp', {}).get('server_url', 'http://localhost:8000')
                self.mcp_adapter = MCPToolAdapter(server_url)
            else:
                self.mcp_adapter = None

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on configuration."""
        llms_config = self.config.get('llms', {})
        from openai import OpenAI
        # Check for OpenAI configuration
        if 'openai' in llms_config:
            api_key = llms_config['openai'].get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key is required in config")
            self.client = OpenAI(api_key=api_key)
            self.llm_type = 'openai'
            
        # Check for DeepSeek configuration
        elif 'deepseek' in llms_config:
            from openai import OpenAI
            print("DeepSeek configuration found")
            api_key = llms_config['deepseek'].get('api_key', 'EMPTY')
            endpoint = llms_config['deepseek'].get('endpoint')
            if not endpoint:
                raise ValueError("DeepSeek endpoint is required in config")
            self.client = OpenAI(
                api_key=api_key,
                base_url=endpoint
            )
            self.llm_type = 'deepseek'
            
        # Check for Ollama configuration
        elif 'ollama' in llms_config:
            from openai import OpenAI
            self.client = OpenAI(
                api_key='EMPTY',
                base_url='http://localhost:11434/v1'
            )
            self.llm_type = 'ollama'
            
        # Check for Local configuration
        elif 'local' in llms_config:
            from openai import OpenAI
            endpoint = llms_config['local'].get('endpoint')
            if not endpoint:
                raise ValueError("Local endpoint is required in config")
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=endpoint
            )
            self.llm_type = 'local'
            
        else:
            raise ValueError("No valid LLM configuration found in config")

    def execute(self, query: str) -> str:
        """
        Execute the agent's main task based on the input query.

        :param query: The input query or task description
        :return: The result of the execution
        """
        try:
            llm_config = self.config.get('llms', {}).get(self.llm_type, {})
            
            print("llm_config", llm_config)
            # Execute the model call
            response = self.client.chat.completions.create(
                model=llm_config.get('model', 'gpt-4'),
                temperature=llm_config.get('temperature', 0.7),
                messages=[
                    {"role": "system", "content": self.config.get('prompt_template', '')},
                    {"role": "user", "content": query}
                ],
                max_tokens=llm_config.get('max_tokens'),
                top_p=llm_config.get('top_p', 1.0),
                frequency_penalty=llm_config.get('frequency_penalty', 0.0),
                presence_penalty=llm_config.get('presence_penalty', 0.0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            raise

    async def initialize(self):
        """Initialize the agent and MCP connection if enabled."""
        if self.is_mcp_enabled and self.mcp_adapter:
            await self.mcp_adapter.initialize()
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        if not self.is_mcp_enabled or not self.mcp_adapter:
            return []
        return await self.mcp_adapter.list_tools()
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a specific MCP tool with given arguments."""
        if not self.is_mcp_enabled or not self.mcp_adapter:
            raise ValueError("MCP tools are not enabled for this agent")
        return await self.mcp_adapter.call_tool(tool_name, arguments=kwargs)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.is_mcp_enabled and self.mcp_adapter:
            await self.mcp_adapter.close()

    def log_message(self, message: str):
        """Logs a message for debugging and tracking purposes."""
        self.logger.info(f"{self.name}: {message}")

    def handle_error(self, error: Exception):
        """Handles errors by logging them."""
        self.logger.error(f"Error in {self.name}: {str(error)}")

class MCPToolAdapter:
    """Adapter class to interact with MCP server tools."""
    
    def __init__(self, server_url: str):
        """
        Initialize the MCP tool adapter.
        
        :param server_url: The URL of the MCP server
        """
        self.server_url = server_url
        self._session = None
        self._sse_client = None
        self._available_tools = None
    
    async def initialize(self):
        """Initialize the MCP client connection."""
        try:
            # Create SSE client and session using context managers
            self._sse_client = sse_client(url=f"{self.server_url}/sse")
            async with self._sse_client as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._available_tools = await session.list_tools()
                    tools_list = [tool.name for tool in self._available_tools.tools]
                    print(f"Initialized MCP connection. Available tools: {tools_list}")
        except Exception as e:
            print(f"Error initializing MCP connection: {str(e)}")
            raise
    
    async def list_tools(self) -> List[str]:
        """Get list of available tools from MCP server."""
        if not self._available_tools:
            await self.initialize()
        return self._available_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server."""
        try:
            print(f"Calling tool: {tool_name} with arguments: {arguments}")
            async with sse_client(url=f"{self.server_url}/sse") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Convert arguments to match the tool's expected format
                    print("tool_name", tool_name)
                    if tool_name == "add":
                        a = int(arguments.get("a", 0))
                        b = int(arguments.get("b", 0))
                        print(f"Calling add with a={a}, b={b}")
                        result = await session.call_tool(tool_name, arguments={"a": a, "b": b})
                        print(f"Add result: {result}")
                        return result
                    elif tool_name == "web_search":
                        query = arguments.get("query", "")
                        print(f"Calling web_search with query={query}")
                        result = await session.call_tool(tool_name, query=query)
                        print(f"Web search result: {result}")
                        return result
                    else:
                        print(f"Calling {tool_name} with arguments: {arguments}")
                        result = await session.call_tool(tool_name, **arguments)
                        print(f"{tool_name} result: {result}")
                        return result
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            print(error_msg)
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"Caused by: {e.__cause__}")
            raise ValueError(error_msg)
    
    async def close(self):
        """Cleanup resources."""
        self._session = None
        self._sse_client = None
        self._available_tools = None
