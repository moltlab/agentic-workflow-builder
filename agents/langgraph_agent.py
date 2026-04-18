from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Dict, Any, List, Union, Optional, TypedDict, Annotated
import operator
import json
import uuid
import os

from utils.logging_utils import get_logger
from db.crud.agent import get_agent_by_id
from db.crud.llm_usage import create_llm_usage_log
from db.crud.interaction import create_interaction
from db.models import ToolExecutionLog, Tool, Transaction
from memory.memory_store import ChatMemory
from config.config_loader import ConfigLoader
from agents.llm_implementations import CustomLLM
from agents.templates.template_loader import AgentTemplateLoader
from .mcp_client import MCPClient
from urllib.parse import urljoin
from opentelemetry import trace
from utils.selectors import rank_files_by_query
from utils.markdown_utils import extract_markdown_points

logger = get_logger('langgraph_agent')
tracer = trace.get_tracer("langgraph_agent")

config_loader = ConfigLoader()
config = config_loader.load_config()

default_mcp_config = config.get('mcp', {})
default_mcp_server_url = default_mcp_config.get('server_url', 'http://localhost:8001')

agent_memories: Dict[str, Dict[str, ChatMemory]] = {}
template_loader = AgentTemplateLoader()

def get_agent_memory(agent_id: str, session_id: str, max_history: int = 10) -> ChatMemory:
    if agent_id not in agent_memories:
        agent_memories[agent_id] = {}
    if session_id not in agent_memories[agent_id]:
        agent_memories[agent_id][session_id] = ChatMemory(max_history=max_history)
    return agent_memories[agent_id][session_id]

class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self, db: Session, transaction_id: uuid.UUID, agent_id: uuid.UUID, tool_usage: List[Dict]):
        self.db, self.transaction_id, self.agent_id, self.tool_usage = db, transaction_id, agent_id, tool_usage
        self.tool_starts, self.tool_log_ids = {}, {}
        # Get user context from transaction
        from utils.user_context import get_user_context_from_transaction
        try:
            self.user_id, self.entity_id = get_user_context_from_transaction(db, transaction_id)
        except ValueError:
            from utils.user_context import resolve_user_entity
            self.user_id, self.entity_id = resolve_user_entity(None)
        # Get user context from transaction
        from utils.user_context import get_user_context_from_transaction
        try:
            self.user_id, self.entity_id = get_user_context_from_transaction(db, transaction_id)
        except ValueError:
            from utils.user_context import resolve_user_entity
            self.user_id, self.entity_id = resolve_user_entity(None)
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any):
        tool_name, run_id = serialized.get("name"), kwargs.get("run_id")
        self.tool_starts[run_id] = datetime.now()
        tool_db = self.db.query(Tool).filter(Tool.name == tool_name).first()
        if not tool_db: return
        try:
            log = ToolExecutionLog(
                transaction_id=self.transaction_id, 
                agent_id=self.agent_id, 
                user_id=self.user_id,
                entity_id=self.entity_id,
                tool_id=tool_db.id, 
                start_time=self.tool_starts[run_id], 
                input_data=input_str
            )
            self.db.add(log); self.db.commit(); self.db.refresh(log)
            self.tool_log_ids[run_id] = log.id
        except Exception as e:
            logger.error(f"DB error: {e}"); self.db.rollback()
    def on_tool_end(self, output: str, **kwargs: Any):
        run_id = kwargs.get("run_id")
        if run_id in self.tool_starts:
            start, log_id = self.tool_starts.pop(run_id), self.tool_log_ids.pop(run_id)
            end = datetime.now()
            duration_ms = int((end - start).total_seconds() * 1000)
            try:
                log = self.db.query(ToolExecutionLog).filter(ToolExecutionLog.id == log_id).first()
                if log:
                    log.end_time, log.duration_ms, log.output_data = end, duration_ms, output
                    self.db.commit()
                    tool_db = self.db.query(Tool).filter(Tool.id == log.tool_id).first()
                    self.tool_usage.append({"tool_id": str(tool_db.id), "tool_name": tool_db.name, "start_time": start, "end_time": end, "duration_ms": duration_ms, "input": log.input_data, "output": output})
            except Exception as e:
                logger.error(f"DB error: {e}"); self.db.rollback()

def _initialize_llm(config_dict: Dict[str, Any]) -> Union[ChatOpenAI, CustomLLM]:
    """
    Initialize the appropriate LLM based on configuration.
    
    :param config_dict: Dictionary containing LLM configuration
    :return: Initialized LLM instance
    """
    llms_config = config_dict.get('llms', {})
    logger.info(f"Initializing LLM with config: {llms_config}")
    
    # Check for OpenAI configuration
    if 'openai' in llms_config:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
        return ChatOpenAI(
            api_key=api_key,
            model_name=llms_config['openai'].get('model', 'gpt-4'),
            temperature=llms_config['openai'].get('temperature', 0.7),
            max_tokens=llms_config['openai'].get('max_tokens', 2000)
        )
    
    # Check for DeepSeek configuration
    elif 'deepseek' in llms_config:
        DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
        llm = CustomLLM(
            model=f"deepseek/{llms_config['deepseek'].get('model', 'default')}",
            base_url=llms_config['deepseek'].get('endpoint', DEEPSEEK_URL),
            api_key=llms_config['deepseek'].get('api_key', "EMPTY"),
            temperature=llms_config['deepseek'].get('temperature', 0.3),
            max_tokens=llms_config['deepseek'].get('max_tokens', 1024)
        )
        return llm
    
    elif 'qwen' in llms_config:
        QWEN_URL = os.getenv("QWEN_URL")
        llm = CustomLLM(
            name="qwen",
            model=f"openai/Qwen/{llms_config['qwen'].get('model', 'Qwen3-32B')}",
            base_url=llms_config['qwen'].get('endpoint', QWEN_URL),
            api_key=llms_config['qwen'].get('api_key', "None"),
            temperature=llms_config['qwen'].get('temperature', 0.3),
            max_tokens=llms_config['qwen'].get('max_tokens', 1024)
        )
        return llm
    
    # Check for Ollama configuration
    elif 'ollama' in llms_config:
        return ChatOpenAI(
            api_key='EMPTY',
            model_name=llms_config['ollama'].get('model', 'llama2'),
            temperature=llms_config['ollama'].get('temperature', 0.7),
            base_url='http://localhost:11434/v1'
        )
    
    # Check for Local configuration
    elif 'local' in llms_config:
        endpoint = llms_config['local'].get('endpoint')
        if not endpoint:
            raise ValueError("Local endpoint is required in config")
        
        return ChatOpenAI(
            api_key='EMPTY',
            model_name=llms_config['local'].get('model', 'local-model'),
            temperature=llms_config['local'].get('temperature', 0.7),
            base_url=endpoint
        )
    
    else:
        raise ValueError("No valid LLM configuration found in config")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def _should_continue(state: AgentState):
    if isinstance(state["messages"][-1], AIMessage) and not state["messages"][-1].tool_calls:
        return "end"
    return "continue"

async def get_and_run_langgraph_agent(db: Session, agent_id: str, query: str, transaction_id: uuid.UUID) -> Dict:
    """
    Get an agent from the database, create a LangGraph instance, and run it.
    
    :param db: Database session
    :param agent_id: ID of the agent in the database
    :param query: Query to run with the agent
    :param transaction_id: UUID of the existing transaction
    :return: Dictionary containing the result and usage information
    """
    general_mcp_client = None
    mcp_client = None
    
    try:
        # Get agent from database
        agent_db = get_agent_by_id(db, agent_id)
        if not agent_db:
            raise ValueError(f"Agent with ID {agent_id} not found")
        
        # Get session_id from transaction
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not transaction:
            raise ValueError(f"Transaction with ID {transaction_id} not found")
        session_id = str(transaction.session_id)
        
        # Convert to dictionary format
        agent_data = {
            'name': agent_db.name,
            'description': agent_db.description,
            'config': agent_db.config,
            'prompt_template': agent_db.prompt_template
        }
        
        # Get memory configuration
        memory_config = agent_data['config'].get('memory', {})
        max_history = memory_config.get('max_history', 10)
        context_window = memory_config.get('context_window', 5)
        
        # Get or create memory store for this agent and session
        memory = get_agent_memory(agent_id, session_id, max_history)
        
        # Add user query to memory
        memory.add_message('user', query)
        
        # Get recent interactions for context
        from db.crud.interaction import list_interactions_by_session_and_agent, get_agent_sessions_and_interactions
        
        context_window = 11
        recent_interactions = list_interactions_by_session_and_agent(db, session_id, uuid.UUID(agent_id), context_window)
        
        paired_interactions = get_agent_sessions_and_interactions(db, agent_id, session_id)
        chat_history = []
        if len(paired_interactions["sessions"]) > 0:
            chat_history.append(paired_interactions["sessions"][0]["interactions"][-10:])
        
        tool_usage = []
        start_time = datetime.now()
        agent_tools = []
        
        # Initialize MCP tools if enabled
        mcp_config = agent_data['config'].get('mcp', {})
        if mcp_config.get('enabled', False):
            mcp_server_url = mcp_config.get('server_url', default_mcp_server_url)
            if mcp_server_url == "":
                mcp_server_url = default_mcp_server_url
            logger.info(f"MCP is enabled. Using server URL: {mcp_server_url}")

            # Initialize MCPClient for Resources and Prompts
            mcp_client_server_base = mcp_server_url.split('/mcp')[0] if '/mcp' in mcp_server_url else mcp_server_url
            sse_mcp_url = urljoin(mcp_client_server_base, 'mcp/sse')
            
            logger.info(f"Connecting MCPClient for resources/prompts at base: {mcp_client_server_base}, SSE: {sse_mcp_url}")
            general_mcp_client = MCPClient(sse_url=sse_mcp_url, server_base_url=mcp_client_server_base)
            try:
                general_mcp_client.connect()
                general_mcp_client.initialize()
                # Add custom tools for resources and prompts
                agent_tools.append(ResourceAccessTool(mcp_client=general_mcp_client))
                agent_tools.append(ListFilesTool(mcp_client=general_mcp_client))
                agent_tools.append(SummarizeTextTool(mcp_client=general_mcp_client))
                agent_tools.append(FindKeywordsTool(mcp_client=general_mcp_client))
                logger.info("Registered custom MCP resource/prompt tools.")
            except RuntimeError as e:
                logger.error(f"Failed to connect or initialize general MCP client: {e}. Proceeding without resource/prompt tools.")

            # Initialize LangChain MCP Client for other MCP Tools
            try:
                # Create MCP client configuration for SSE connection
                mcp_client_config = {
                    "mcp_server": {
                        "transport": "sse",
                        "url": f"{mcp_server_url}/sse"
                    }
                }
                
                # Initialize the MCP client
                mcp_client = MultiServerMCPClient(mcp_client_config)
                
                # Get tools from MCP server
                mcp_tools_from_client = await mcp_client.get_tools()
                logger.info(f"Retrieved {len(mcp_tools_from_client)} tools from LangChain MCP Client")
                
                if mcp_tools_from_client:
                    configured_tool_names = [tool['name'] for tool in mcp_config.get('tools', [])]
                    if configured_tool_names:
                        filtered_mcp_tools = [tool for tool in mcp_tools_from_client if tool.name in configured_tool_names]
                    else:
                        filtered_mcp_tools = mcp_tools_from_client
                    
                    # Add MCP tools directly (same as crew_agent.py)
                    agent_tools.extend(filtered_mcp_tools)
                    logger.info(f"Registered {len(filtered_mcp_tools)} tools from LangChain MCP Client: {[tool.name for tool in filtered_mcp_tools]}")
                else:
                    logger.warning("No tools found from LangChain MCP Client.")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain MCP Client or fetch tools: {e}. Proceeding without these MCP tools.")
        else:
            logger.info("MCP is not enabled for this agent.")
        
        # Initialize LLM based on configuration
        try:
            llm = _initialize_llm(agent_data['config'])
            logger.info("LLM initialized with model: %s", llm.model_name)
            
            # Store LLM configuration for logging
            llm_usage = {
                "model_name": llm.model_name,
                "model_provider": next((k for k in ["openai", "deepseek", "qwen", "ollama"] if k in agent_data['config'].get('llms', {})), "local"),
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "start_time": datetime.now()
            }
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None
        
        # Load backstory and output format from templates
        backstory = template_loader.get_backstory(agent_db.name)
        if backstory:
            backstory = backstory + f"""
            Recent conversation history:
            {chat_history}.
            """
        else:
            # Use default backstory if no template exists
            backstory = f"""You are {agent_data['name']}, {agent_data['description']}   
            Recent conversation history:
            {chat_history}.

            Please use the output from the tool to answer the user's query in a proper sentence.
            If the tool output is not relevant to the user's query, please use the LLM to answer the query.
            """
        
        # Add custom tool descriptions to backstory if general_mcp_client was successfully initialized
        if general_mcp_client:
            backstory = backstory + f"""

            You have access to a 'resource_reader' tool to read content from files (e.g., file:///shared_files/your_document.pdf), a 'list_files' tool to see available resources, 'summarize_text' to summarize content, and 'find_keywords' to extract keywords. Prioritize using these tools for file-related queries.
            """
        
        # Add MCP tool descriptions to backstory
        logger.info(f"Total agent tools: {len(agent_tools)}")
        for tool in agent_tools:
            logger.info(f"Tool: {getattr(tool, 'name', 'NO_NAME')} - {getattr(tool, 'description', 'NO_DESC')[:100]}")
        
        if agent_tools:
            tool_descriptions = []
            for tool in agent_tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            if tool_descriptions:
                backstory = backstory + f"""

            Available tools:
            {chr(10).join(tool_descriptions)}
            
            CRITICAL INSTRUCTION: You MUST use the appropriate tool to answer the user's query. Do not make up or hallucinate information.
            When you need to use a tool, respond with EXACTLY this format: TOOL_CALL: {{"tool_name": "tool_name", "args": {{"param1": "value1"}}}}
            
            For mobile number verification, you MUST use the mobile_account_lookup_tool with the mobile number and a referenceId.
            Example: TOOL_CALL: {{"tool_name": "mobile_account_lookup_tool", "args": {{"mobile_number": "9948521610", "referenceId": "12345"}}}}
            
            For document reading, you MUST use the skyc_read_doc_tool with the image source and referenceId.
            Example: TOOL_CALL: {{"tool_name": "skyc_read_doc_tool", "args": {{"image_src": "test_images/readdoc_pan.png", "referenceId": "pan_test"}}}}
            
            For listing resources, you MUST use the list_files tool.
            Example: TOOL_CALL: {{"tool_name": "list_files", "args": {{}}}}
            
            Do not provide any response without first calling the appropriate tool.
            """

        # Determine task description based on query and available resources
        task_description = query
        # Only attempt file-related logic if general_mcp_client was successfully initialized
        if general_mcp_client and ("notes" in query.lower() or "project notes" in query.lower() or "files" in query.lower() or "documents" in query.lower()):
            try:
                files = general_mcp_client.list_project_files()
                if files:
                    relevant_files = rank_files_by_query(files, query)
                    if relevant_files:
                        logger.info(f"Identified relevant files: {relevant_files}")
                        file_content_chunks = []
                        for fname in relevant_files:
                            uri = f"file:///{fname}"
                            content = general_mcp_client.read_resource_text(uri)
                            if content:
                                file_content_chunks.append(f"From {fname}:\n" + content)
                        
                        if file_content_chunks:
                            joined_content = "\n\n".join(file_content_chunks)
                            if "summarize" in query.lower():
                                task_description = f"Summarize the following content based on the user's query: {query}\n\nContent:\n{joined_content}"
                            elif "keywords" in query.lower():
                                task_description = f"Extract keywords from the following content based on the user's query: {query}\n\nContent:\n{joined_content}"
                            else:
                                task_description = f"Answer the user's query using the following content: {query}\n\nContent:\n{joined_content}"
                        else:
                            logger.warning("Could not read content from identified relevant files.")
                            task_description = query
                    else:
                        logger.info("No relevant files found based on query.")
                        task_description = query
                else:
                    logger.info("No project files available from MCP server.")
                    task_description = query
            except Exception as e:
                logger.error(f"Error processing file-related query: {e}")
                task_description = query

        # Create system prompt - escape any curly braces in backstory
        escaped_backstory = backstory.replace("{", "{{").replace("}", "}}")
        system_prompt = f"{escaped_backstory}\n\nCurrent task: {task_description}"

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        
        # Convert all tools to a format that can be bound to the LLM
        def convert_tools_for_binding(tools):
            converted_tools = []
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    try:
                        # Create a simple tool definition for binding
                        tool_def = {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                        
                        # Try to get parameters from tool's args_schema if available
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            try:
                                schema = tool.args_schema.model_json_schema()
                                if schema and 'properties' in schema and schema['properties']:
                                    tool_def["parameters"]["properties"] = schema['properties']
                                if schema and 'required' in schema and schema['required']:
                                    tool_def["parameters"]["required"] = schema['required']
                            except Exception as e:
                                logger.warning(f"Could not extract schema from tool {tool.name}: {e}")
                        
                        # Validate the tool definition before adding
                        if tool_def["parameters"] and isinstance(tool_def["parameters"], dict):
                            converted_tools.append(tool_def)
                        else:
                            logger.warning(f"Skipping tool {tool.name} due to invalid parameters")
                    except Exception as e:
                        logger.warning(f"Could not convert tool {tool.name}: {e}")
            return converted_tools

        # Bind tools to LLM using LangChain MCP adapters
        if agent_tools:
            try:
                agent_runnable = prompt | llm.bind_tools(agent_tools)
            except Exception as e:
                logger.error(f"Failed to bind tools: {e}")
                agent_runnable = prompt | llm
        else:
            agent_runnable = prompt | llm

        def agent_node(state: AgentState):
            return {"messages": [agent_runnable.invoke(state)]}

        # Create workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        tool_node = ToolNode(agent_tools)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", _should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        app = workflow.compile()
        
        # Create callback handler
        callback_handler = AgentCallbackHandler(db, transaction_id, uuid.UUID(agent_id), tool_usage)
        
        # Get chat history
        raw_history = memory.get_history()
        chat_history = [HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content']) for msg in raw_history]
        
        # Execute the workflow with OpenTelemetry tracing
        with tracer.start_as_current_span("langgraph-agent-kickoff") as span:
            # Add span attributes that the Streamlit dashboard expects
            span.set_attribute("gen_ai.agent.name", agent_data['name'])
            span.set_attribute("agent.id", agent_id)
            span.set_attribute("session.id", session_id)
            
            # Add prompt information for dashboard search functionality
            task_prompt = f"Current Task: {query}\n\nAgent: {agent_data['name']}\nRole: {agent_data['description']}\nSession: {session_id}"
            span.set_attribute("gen_ai.prompt", task_prompt)
            
            # Add agent tools information for dashboard display
            if agent_tools:
                tools_list = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent_tools]
                span.set_attribute("gen_ai.agent.tools", ", ".join(tools_list))
            
            # Add configuration details
            span.set_attribute("query.length", len(query))
            
            logger.info(f"Starting LangGraph workflow with span ID: {hex(span.get_span_context().span_id)}")
            logger.info(f"LangGraph workflow trace ID: {hex(span.get_span_context().trace_id)}")
            
            # Execute the workflow
            start_time_workflow = datetime.now()
            final_state = await app.ainvoke({"messages": chat_history}, config={"callbacks": [callback_handler]})
            end_time_workflow = datetime.now()
            
            final_output = final_state["messages"][-1].content
            
            # Add completion information for dashboard display
            completion_text = f"Final Answer: {final_output}"
            span.set_attribute("gen_ai.completion", completion_text)
            
            # Add result metadata to span
            execution_time_ms = int((end_time_workflow - start_time_workflow).total_seconds() * 1000)
            span.set_attribute("langgraph.execution_time_ms", execution_time_ms)
            span.set_attribute("langgraph.result_length", len(final_output))
            span.set_attribute("langgraph.result_type", type(final_output).__name__)
            
            # Log additional details
            logger.info(f"LangGraph workflow finished in {execution_time_ms}ms")
            logger.info(f"Result length: {len(final_output)} characters")
            logger.info(f"Span recording: {span.is_recording()}")
        
        # Add agent response to memory
        memory.add_message('assistant', final_output)
        
        # Get user context from transaction for interactions
        from utils.user_context import get_user_context_from_transaction
        try:
            user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
        except ValueError:
            user_id, entity_id = None, None
        
        # Create interactions
        create_interaction(db, {
            "session_id": session_id,
            "sender": "user",
            "message": query,
            "timestamp": datetime.now(),
            "agent_id": uuid.UUID(agent_id),
            "message_metadata": {}
        }, user_id=user_id, entity_id=entity_id)
        
        create_interaction(db, {
            "session_id": session_id,
            "sender": "assistant",
            "message": final_output,
            "timestamp": datetime.now(),
            "agent_id": uuid.UUID(agent_id),
            "message_metadata": {}
        }, user_id=user_id, entity_id=entity_id)

        # Add tool output to agents memory for specific agents
        if (agent_db.name == "Underwriting" or "Retail" in agent_db.name):
            tool_execution_log = db.query(ToolExecutionLog).filter(ToolExecutionLog.transaction_id == transaction_id, ToolExecutionLog.agent_id == uuid.UUID(agent_id)).order_by(ToolExecutionLog.id.desc()).first()
            if tool_execution_log:
                logger.info(f"Creating interaction for tool output : {agent_db.name}")
                # Get user context from transaction (reuse from above if available)
                if 'user_id' not in locals() or 'entity_id' not in locals():
                    from utils.user_context import get_user_context_from_transaction
                    try:
                        user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
                    except ValueError:
                        user_id, entity_id = None, None
                create_interaction(db, {
                    "session_id": session_id,
                    "sender": "assistant",
                    "message": str(tool_execution_log.output_data),
                    "timestamp": datetime.now(),
                    "agent_id": uuid.UUID(agent_id),
                    "message_metadata": {}
                }, user_id=user_id, entity_id=entity_id)
                memory.add_message('assistant', str(tool_execution_log.output_data))
        
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Update LLM usage with end time
        llm_usage["end_time"] = end_time
        llm_usage["duration_ms"] = duration_ms
        
        # Convert datetime objects to ISO format strings
        llm_usage = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in llm_usage.items()}
        
        # Serialize datetime objects in tool_usage
        serialized_tool_usage = []
        for tool in tool_usage:
            serialized_tool = {
                "tool_id": tool["tool_id"],
                "tool_name": tool["tool_name"],
                "start_time": tool["start_time"].isoformat() if isinstance(tool["start_time"], datetime) else tool["start_time"],
                "end_time": tool["end_time"].isoformat() if isinstance(tool["end_time"], datetime) else tool["end_time"],
                "duration_ms": tool["duration_ms"],
                "input": tool["input"],
                "output": tool["output"]
            }
            serialized_tool_usage.append(serialized_tool)
        
        # Create LLM usage log
        create_llm_usage_log(db, {
            "transaction_id": transaction_id,
            "agent_id": uuid.UUID(agent_id),
            "start_time": llm_usage.get("start_time"),
            "end_time": llm_usage.get("end_time"),
            "duration_ms": llm_usage.get("duration_ms"),
            "model_name": llm_usage.get("model_name"),
            "model_provider": llm_usage.get("model_provider"),
            "temperature": llm_usage.get("temperature"),
            "max_tokens": llm_usage.get("max_tokens"),
            "total_tokens_used": llm_usage.get("total_tokens", 0),
            "response_latency_ms": llm_usage.get("latency_ms", 0),
            "input_data": {"query": query},
            "output_data": {"result": final_output}
        })
        
        # Return result with usage information and chat history
        return {
            "result": final_output,
            "tool_usage": serialized_tool_usage,
            "llm_usage": llm_usage,
            "total_duration_ms": duration_ms,
            "total_tools_used": len(serialized_tool_usage),
            "chat_history": memory.get_history()
        }
        
    except Exception as e:
        logger.error(f"Error running langgraph agent: {str(e)}")
        raise
    finally:
        # Cleanup MCP clients
        if general_mcp_client:
            general_mcp_client.stop()
            logger.info("General MCP client stopped and cleaned up.")

async def get_and_run_langgraph_with_agents(db: Session, agent_ids: List[str], query: str, transaction_id: Optional[uuid.UUID] = None) -> Dict:
    """
    Run multiple LangGraph agents in sequence.
    
    :param db: Database session
    :param agent_ids: List of agent IDs to run
    :param query: Initial query
    :param transaction_id: Optional transaction ID
    :return: Dictionary containing the final result
    """
    try:
        current_input = query
        final_result = ""
        
        # Get session_id from transaction if provided
        if transaction_id:
            transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
            if not transaction:
                raise ValueError(f"Transaction with ID {transaction_id} not found")
            session_id = str(transaction.session_id)
        else:
            session_id = str(uuid.uuid4())  # Generate new session ID if none provided
        
        for agent_id in agent_ids:
            # Create a new transaction for each agent
            agent_transaction_id = uuid.uuid4()
            result_dict = await get_and_run_langgraph_agent(db, agent_id, current_input, agent_transaction_id)
            current_input = result_dict['result']
            final_result = current_input
        
        return {"result": final_result}
        
    except Exception as e:
        logger.error(f"Error in get_and_run_langgraph_with_agents: {str(e)}")
        raise

class ResourceAccessTool(BaseTool):
    name: str = "resource_reader"
    description: str = (
        "Reads content from a specified resource on the MCP server. "
        "Use this to access files or data lists. The input must be the full resource URI, "
        "for example: 'file:///shared_files/report.pdf' or 'resource://files/list'."
    )
    mcp_client: MCPClient

    def _run(self, uri: str) -> str:
        """The tool's execution method."""
        try:
            return self.mcp_client.read_resource_text(uri=uri)
        except Exception as e:
            logger.error(f"ResourceAccessTool failed for URI '{uri}': {e}")
            return f"An error occurred while trying to read the resource: {e}"

class ListFilesTool(BaseTool):
    name: str = "list_files"
    description: str = (
        "Lists all available project files from the MCP server's shared resources. "
        "Use this to see what files are accessible to the agent. No input required."
    )
    mcp_client: MCPClient

    def _run(self) -> str:
        """The tool's execution method."""
        try:
            files = self.mcp_client.list_project_files()
            file_list = '\n'.join(files) if files else 'None'
            return f"Available files: {file_list}"
        except Exception as e:
            logger.error(f"ListFilesTool failed: {e}")
            return f"An error occurred while trying to list files: {e}"

class SummarizeTextTool(BaseTool):
    name: str = "summarize_text"
    description: str = (
        "Summarizes a given text using an MCP prompt. "
        "Input should be the text content to be summarized."
    )
    mcp_client: MCPClient

    def _run(self, text: str) -> str:
        """The tool's execution method."""
        try:
            return self.mcp_client.call_tool("summarize_text", text=text)
        except Exception as e:
            logger.error(f"SummarizeTextTool failed: {e}")
            return f"An error occurred while trying to summarize text: {e}"

class FindKeywordsTool(BaseTool):
    name: str = "find_keywords"
    description: str = (
        "Extracts keywords from a given text using an MCP prompt. "
        "Input should be the text content from which to extract keywords."
    )
    mcp_client: MCPClient

    def _run(self, text: str) -> str:
        """The tool's execution method."""
        try:
            return self.mcp_client.call_tool("find_keywords", text=text)
        except Exception as e:
            logger.error(f"FindKeywordsTool failed: {e}")
            return f"An error occurred while trying to find keywords: {e}"