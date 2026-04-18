"""
Configuration-driven LangGraph workflows from JSON (agents + graph definition).

Used by the workflow builder API; MCP tools and LLM are wired from environment.
"""

import os
import json
from typing import TypedDict, Dict, Any, Callable

from typing_extensions import Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.shared.mcp_tools_langchain import LangChainMCPToolsManager


class ConfigDrivenWorkflowBuilder:
    """Builds LangGraph workflows from JSON configurations."""

    def __init__(self, agents_config_path: str, workflow_config_path: str):
        self.agents_config_path = agents_config_path
        self.workflow_config_path = workflow_config_path
        self.agents_config = None
        self.workflow_config = None
        self.llm = None
        self.tools_registry = {}
        self.agent_functions = {}

    def load_configs(self):
        """Load agent and workflow configurations."""
        with open(self.agents_config_path, "r") as f:
            self.agents_config = json.load(f)

        with open(self.workflow_config_path, "r") as f:
            self.workflow_config = json.load(f)

        print(f"✓ Loaded agent configurations: {len(self.agents_config)} agents")
        print(f"✓ Loaded workflow configuration: {self.workflow_config['workflow_name']}")

    def initialize_llm(self):
        """Initialize the LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=api_key)
        print("✓ Initialized LLM")

    async def load_tools(self):
        """Load tools from MCP server."""
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
        print(f"Connecting to MCP server at: {mcp_server_url}")

        mcp_config = {"enabled": True, "server_url": mcp_server_url, "tools": []}

        try:
            mcp_manager = LangChainMCPToolsManager(mcp_config)
            mcp_tools = await mcp_manager.initialize_langchain_mcp_tools()

            if not mcp_tools:
                print("⚠ No tools returned from MCP server")
                return

            for tool in mcp_tools:
                tool_name = getattr(tool, "name", None)
                if tool_name:
                    self.tools_registry[tool_name] = tool

            print(f"✓ Loaded {len(self.tools_registry)} tools from MCP server")

        except Exception as e:
            print(f"⚠ Failed to load tools from MCP server: {e}")

    def _auto_extract_prompt_variables(self, prompt_template: str, state: Dict) -> Dict:
        """Map {variables} in prompt_template to state / messages."""
        import re

        variable_pattern = r"\{(\w+)\}"
        variables_needed = re.findall(variable_pattern, prompt_template)

        variables = {}
        state_schema = self.workflow_config["state_schema"]
        messages = state.get("messages", [])

        for var_name in variables_needed:
            if var_name in state_schema:
                variables[var_name] = state.get(var_name, "")
            elif var_name in [
                "information",
                "answer",
                "result",
                "response",
                "data",
                "findings",
            ]:
                if messages:
                    last_msg = messages[-1]
                    variables[var_name] = getattr(last_msg, "content", "")
                else:
                    variables[var_name] = ""
            elif var_name in ["query", "question", "request", "topic"]:
                variables[var_name] = state.get("query", state.get("question", ""))
            elif var_name in ["context", "document", "docs", "background"]:
                variables[var_name] = state.get("context", state.get("document", ""))
            elif var_name in ["user_name", "name", "user"]:
                variables[var_name] = state.get("user_name", state.get("name", ""))
            else:
                variables[var_name] = state.get(var_name, "")
                if not variables[var_name]:
                    print(f"⚠ Variable '{var_name}' not found in state. Using empty string.")

        return variables

    def create_state_class(self):
        """Dynamically create state class from config."""
        state_schema = self.workflow_config["state_schema"]

        fields = {}
        for field_name, field_type in state_schema.items():
            if field_type == "list":
                fields[field_name] = Annotated[list, operator.add]
            elif field_type == "str":
                fields[field_name] = str
            elif field_type == "dict":
                fields[field_name] = dict
            else:
                fields[field_name] = Any

        WorkflowState = TypedDict("WorkflowState", fields)
        return WorkflowState

    def create_agent_function(self, agent_id: str, agent_config: Dict[str, Any]) -> Callable:
        """Create an agent function from configuration."""
        agent_type = agent_config["type"]

        if agent_type == "simple":

            def simple_agent(state):
                output = {}
                for field in agent_config["output_fields"]:
                    output[field] = state.get(field, "")
                return output

            return simple_agent

        if agent_type == "llm":
            prompt_template = agent_config["prompt"]
            output_fields = agent_config["output_fields"]

            def llm_agent(state):
                prompt_vars = self._auto_extract_prompt_variables(prompt_template, state)
                prompt = prompt_template.format(**prompt_vars)
                messages = state.get("messages", [])

                if messages:
                    current_query = state.get("query", "")
                    messages_list = list(messages)
                    if (
                        messages_list
                        and isinstance(messages_list[-1], HumanMessage)
                        and messages_list[-1].content.strip() == current_query.strip()
                    ):
                        messages_with_prompt = messages_list[:-1] + [HumanMessage(content=prompt)]
                    else:
                        messages_with_prompt = messages_list + [HumanMessage(content=prompt)]
                    response = self.llm.invoke(messages_with_prompt)
                else:
                    response = self.llm.invoke([HumanMessage(content=prompt)])

                response_text = response.content.strip().upper()
                if any(
                    marker in response_text
                    for marker in ["UNABLE_TO_ANSWER", "NEED_MORE_INFO", "INSUFFICIENT_INFO"]
                ):
                    print(
                        f"  ⚠ Agent indicated insufficient information with following response: {response_text}"
                    )
                    return {}

                output = {}
                for field in output_fields:
                    if field == "messages":
                        output["messages"] = [response]
                    else:
                        output[field] = response.content.strip()

                return output

            return llm_agent

        if agent_type == "llm_with_tools":
            prompt_template = agent_config["prompt"]
            tool_names = agent_config.get("tools", [])
            tools = [
                self.tools_registry[tool_name]
                for tool_name in tool_names
                if tool_name in self.tools_registry
            ]

            if not tools:
                print(f"⚠ No tools found for agent {agent_id}")
                return None

            llm_with_tools = self.llm.bind_tools(tools)

            def tool_agent(state):
                messages = state.get("messages", [])
                if not messages:
                    prompt_vars = self._auto_extract_prompt_variables(prompt_template, state)
                    prompt = prompt_template.format(**prompt_vars)
                    messages = [HumanMessage(content=prompt)]
                response = llm_with_tools.invoke(messages)
                return {"messages": messages + [response]}

            return tool_agent

        return None

    def create_routers(self):
        """Create router functions from workflow config."""
        routers = {}

        def check_messages_router(state):
            messages = state.get("messages", [])
            if not messages:
                return "no_messages"
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                return "has_messages"
            return "no_messages"

        routers["check_messages"] = check_messages_router

        def check_tool_calls_router(state):
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    return "has_tool_calls"
            return "no_tool_calls"

        routers["check_tool_calls"] = check_tool_calls_router

        return routers

    async def build_workflow(self):
        """Build the workflow graph from configuration."""
        WorkflowState = self.create_state_class()
        graph = StateGraph(WorkflowState)
        routers = self.create_routers()

        for node_config in self.workflow_config["nodes"]:
            node_id = node_config["id"]

            if node_config.get("type") == "tool_node":
                tool_names = node_config.get("tools", [])
                tools = [self.tools_registry[name] for name in tool_names if name in self.tools_registry]
                tool_node = ToolNode(tools)
                graph.add_node(node_id, tool_node)
            elif "agent" in node_config:
                agent_name = node_config["agent"]
                agent_config = self.agents_config[agent_name]
                agent_func = self.create_agent_function(agent_name, agent_config)

                if agent_func:
                    graph.add_node(node_id, agent_func)
                    print(f"  ✓ Created node: {node_id} ({agent_config['type']})")

        for edge_config in self.workflow_config["edges"]:
            from_node = edge_config["from"]
            to_node = edge_config["to"]

            if from_node == "START":
                from_node = START
            if to_node == "END":
                to_node = END

            if to_node == "conditional":
                condition_config = edge_config["condition"]
                condition_type = condition_config["type"]
                routes = condition_config["routes"]
                router_func = routers.get(condition_type)
                if router_func:
                    graph.add_conditional_edges(from_node, router_func, routes)
                    print(f"  ✓ Added conditional edge: {from_node} -> {condition_type}")
            else:
                graph.add_edge(from_node, to_node)
                print(f"  ✓ Added edge: {from_node} -> {to_node}")

        app = graph.compile()
        print("\n✓ Workflow graph compiled successfully")

        return app
