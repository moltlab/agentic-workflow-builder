"""
Hierarchical Workflow Builder for Database-Driven Workflows

Builds LangGraph workflows from database-loaded configurations.
This is a refactored version of ConfigDrivenWorkflowBuilder that works
directly with in-memory config dictionaries instead of JSON files.
"""

import os
import asyncio
from typing import TypedDict, Dict, Any, Callable
from typing_extensions import Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.shared.mcp_tools_langchain import LangChainMCPToolsManager
from utils.logging_utils import get_logger

logger = get_logger('hierarchical_workflow_builder')


class HierarchicalWorkflowBuilder:
    """Builds LangGraph workflows from in-memory database configurations"""
    
    def __init__(self, agents_config: Dict[str, Any], workflow_config: Dict[str, Any]):
        """
        Initialize the workflow builder with in-memory configurations.
        
        Args:
            agents_config: Dictionary mapping agent names to their configurations
            workflow_config: Dictionary containing workflow structure (nodes, edges, state_schema)
        """
        self.agents_config = agents_config or {}
        self.workflow_config = workflow_config or {}
        self.llm = None
        self.tools_registry = {}
        self.agent_functions = {}
    
    def initialize_llm(self):
        """Initialize the LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    
    async def load_tools(self):
        """Load tools from MCP server"""
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
        
        mcp_config = {
            "enabled": True,
            "server_url": mcp_server_url,
            "tools": []
        }
        
        try:
            mcp_manager = LangChainMCPToolsManager(mcp_config)
            mcp_tools = await mcp_manager.initialize_langchain_mcp_tools()
            
            if not mcp_tools:
                return
            
            for tool in mcp_tools:
                tool_name = getattr(tool, 'name', None)
                if tool_name:
                    self.tools_registry[tool_name] = tool
        
        except Exception as e:
            logger.warning(f"Failed to load tools from MCP server: {e}")
    
    def _auto_extract_prompt_variables(self, prompt_template: str, state: Dict) -> Dict:
        """
        Automatically extract variables from prompt template and map to state.
        
        Smart mapping rules:
        1. If variable name exists in state schema → use state field
        2. Common patterns:
           - {information}, {answer}, {result}, {response} → messages[-1].content
           - {query}, {question}, {request} → state.query
           - {context}, {document}, {docs} → state.context
           - {user_name}, {name} → state.user_name
        3. Fallback: Check if field exists in state
        """
        import re
        
        # Extract all {variable} placeholders from prompt
        variable_pattern = r'\{(\w+)\}'
        variables_needed = re.findall(variable_pattern, prompt_template)
        
        variables = {}
        state_schema = self.workflow_config.get('state_schema', {})
        messages = state.get('messages', [])
        
        for var_name in variables_needed:
            # Rule 1: Direct match to state schema
            if var_name in state_schema:
                variables[var_name] = state.get(var_name, "")
            
            # Rule 2: Common patterns for message content
            elif var_name in ['information', 'answer', 'result', 'response', 'data', 'findings']:
                # Get content from last message
                if messages:
                    last_msg = messages[-1]
                    variables[var_name] = getattr(last_msg, 'content', '')
                else:
                    variables[var_name] = ""
            
            # Rule 3: Query patterns
            elif var_name in ['query', 'question', 'request', 'topic']:
                variables[var_name] = state.get('query', state.get('question', ''))
            
            # Rule 4: Context patterns
            elif var_name in ['context', 'document', 'docs', 'background']:
                variables[var_name] = state.get('context', state.get('document', ''))
            
            # Rule 5: User patterns
            elif var_name in ['user_name', 'name', 'user']:
                variables[var_name] = state.get('user_name', state.get('name', ''))
            
            # Fallback: Try to find in state directly
            else:
                variables[var_name] = state.get(var_name, "")
                if not variables[var_name]:
                    logger.warning(f"Variable '{var_name}' not found in state. Using empty string.")
        
        return variables
    
    def create_state_class(self):
        """Dynamically create state class from config"""
        state_schema = self.workflow_config.get('state_schema', {})
        
        # Build TypedDict fields
        fields = {}
        for field_name, field_type in state_schema.items():
            if field_type == "list":
                # Lists should accumulate with operator.add
                fields[field_name] = Annotated[list, operator.add]
            elif field_type == "str":
                fields[field_name] = str
            elif field_type == "dict":
                fields[field_name] = dict
            else:
                fields[field_name] = Any
        
        # Create TypedDict dynamically
        WorkflowState = TypedDict('WorkflowState', fields)
        return WorkflowState
    
    def create_agent_function(self, agent_name: str, agent_config: Dict[str, Any]) -> Callable:
        """Create an agent function from configuration"""
        agent_type = agent_config.get('type')
        
        if agent_type == "simple":
            # Simple pass-through agent (like master)
            def simple_agent(state):
                output = {}
                for field in agent_config.get('output_fields', []):
                    output[field] = state.get(field, "")
                return output
            return simple_agent
        
        elif agent_type == "llm":
            # LLM-based agent without tools
            prompt_template = agent_config.get('prompt', '')
            output_fields = agent_config.get('output_fields', ['messages'])
            
            def llm_agent(state):
                # Auto-extract variables from prompt template
                prompt_vars = self._auto_extract_prompt_variables(prompt_template, state)
                
                # Format prompt
                prompt = prompt_template.format(**prompt_vars)
                
                # Get existing messages from state (conversation history)
                messages = state.get("messages", [])
                
                # If there's conversation history, use it and replace/add the formatted prompt
                if messages:
                    current_query = state.get("query", "")
                    messages_list = list(messages)
                    
                    # If last message is a HumanMessage with the current query, replace it
                    if (messages_list and 
                        isinstance(messages_list[-1], HumanMessage) and
                        messages_list[-1].content.strip() == current_query.strip()):
                        # Replace last message with formatted prompt
                        messages_with_prompt = messages_list[:-1] + [HumanMessage(content=prompt)]
                    else:
                        # Append formatted prompt
                        messages_with_prompt = messages_list + [HumanMessage(content=prompt)]
                    
                    response = self.llm.invoke(messages_with_prompt)
                else:
                    # No history, just use the prompt
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Generic check: If LLM indicates it cannot answer, return empty to trigger alternative path
                response_text = response.content.strip().upper()
                if any(marker in response_text for marker in ["UNABLE_TO_ANSWER", "NEED_MORE_INFO", "INSUFFICIENT_INFO"]):
                    logger.warning(f"  ⚠ Agent indicated insufficient information with following response: {response_text}")
                    return {}
                
                # Build output based on output_fields
                output = {}
                for field in output_fields:
                    if field == 'messages':
                        output['messages'] = [response]
                    else:
                        # For other fields, use response content
                        output[field] = response.content.strip()
                
                return output
            
            return llm_agent
        
        elif agent_type == "llm_with_tools":
            # LLM-based agent with tools
            prompt_template = agent_config.get('prompt', '')
            tool_names = agent_config.get('tools', [])
            
            # Get tools from registry
            tools = [self.tools_registry[tool_name] for tool_name in tool_names if tool_name in self.tools_registry]
            
            if not tools:
                logger.warning(f"No tools found for agent '{agent_name}' (requested: {tool_names})")
                return None
            
            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(tools)
            
            def tool_agent(state):
                messages = state.get("messages", [])
                
                # Add user query if no messages yet
                if not messages:
                    # Auto-extract variables from prompt template
                    prompt_vars = self._auto_extract_prompt_variables(prompt_template, state)
                    prompt = prompt_template.format(**prompt_vars)
                    messages = [HumanMessage(content=prompt)]
                
                # Invoke LLM with tools
                response = llm_with_tools.invoke(messages)
                
                # Return updated messages with agent response
                return {"messages": messages + [response]}
            
            return tool_agent
        
        logger.warning(f"Unknown agent type '{agent_type}' for agent '{agent_name}'")
        return None
    
    def create_routers(self):
        """Create router functions from workflow config"""
        routers = {}
        
        # Check messages router
        def check_messages_router(state):
            messages = state.get("messages", [])
            if not messages:
                return "no_messages"
            
            # Check if the last message is an AIMessage (from information_retriever)
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                # information_retriever successfully provided an answer
                return "has_messages"
            else:
                # information_retriever didn't add a response (returned empty dict)
                return "no_messages"
        
        routers['check_messages'] = check_messages_router
        
        # Check tool calls router
        def check_tool_calls_router(state):
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    return "has_tool_calls"
            return "no_tool_calls"
        
        routers['check_tool_calls'] = check_tool_calls_router
        
        return routers
    
    async def build_workflow(self):
        """Build the workflow graph from configuration"""
        # Create state class
        WorkflowState = self.create_state_class()
        
        # Create graph
        graph = StateGraph(WorkflowState)
        
        # Create routers
        routers = self.create_routers()
        
        nodes_added = 0
        node_names = []
        # Create agent functions and add nodes
        for node_config in self.workflow_config.get('nodes', []):
            node_id = node_config.get('id')
            if not node_id:
                continue
            
            if node_config.get('type') == 'tool_node':
                # Create tool node
                tool_names = node_config.get('tools', [])
                tools = [self.tools_registry[name] for name in tool_names if name in self.tools_registry]
                tool_node = ToolNode(tools)
                graph.add_node(node_id, tool_node)
                node_names.append(f"{node_id}(tool)")
                nodes_added += 1
            elif 'agent' in node_config:
                # Create agent node
                agent_id = node_config['agent']
                # Look up agent by ID (agents_config uses agent IDs as keys)
                agent_config = self.agents_config.get(agent_id)
                if not agent_config:
                    logger.warning(f"Agent '{agent_id}' not found in agents_config for node '{node_id}'")
                    continue
                
                agent_func = self.create_agent_function(agent_id, agent_config)
                
                if agent_func:
                    graph.add_node(node_id, agent_func)
                    node_names.append(f"{node_id}({agent_config.get('type', 'unknown')})")
                    nodes_added += 1
        
        edges_added = 0
        edge_list = []
        # Add edges
        for edge_config in self.workflow_config.get('edges', []):
            from_node = edge_config.get('from')
            to_node = edge_config.get('to')
            
            if not from_node or not to_node:
                continue
            
            # Handle START and END special nodes
            if from_node == "START":
                from_node = START
            if to_node == "END":
                to_node = END
            
            if to_node == "conditional":
                # Conditional edge
                condition_config = edge_config.get('condition', {})
                condition_type = condition_config.get('type')
                routes = condition_config.get('routes', {})
                
                router_func = routers.get(condition_type)
                if router_func:
                    graph.add_conditional_edges(from_node, router_func, routes)
                    edge_list.append(f"{from_node} -> [{condition_type}]")
                    edges_added += 1
            else:
                # Regular edge
                graph.add_edge(from_node, to_node)
                edge_list.append(f"{from_node} -> {to_node}")
                edges_added += 1
        
        # Compile graph
        app = graph.compile()
        
        # Log workflow structure summary
        logger.info(f"Workflow Structure: {nodes_added} nodes ({', '.join(node_names)}), {edges_added} edges")
        
        return app

