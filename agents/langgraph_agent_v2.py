"""
Example of refactored langgraph_agent.py using generic modules.
This shows how to maximize reuse of shared functionality across frameworks.
"""

import uuid
import operator
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Annotated, Optional
from langchain_core import messages
from sqlalchemy.orm import Session

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from opentelemetry import trace

# Import generic modules from agents.shared - MAXIMUM REUSE!
from agents.shared import (
    LLMFactory,
    log_agent_step_langgraph,
    create_agent_execution_span,
    log_llm_usage,
    log_agent_interactions,
    serialize_tool_usage,
    TemplateManager,
    AgentDatabaseManager,
    build_multimodal_content,
)

# Import LangChain-specific MCP tools (PROPER SEPARATION!)
from agents.shared.mcp_tools_langchain import LangChainMCPToolsManager

from utils.logging_utils import get_logger

logger = get_logger('langgraph_agent_v2')
tracer = trace.get_tracer("langgraph_agent_v2")


async def create_langgraph_agent_from_config(
    db: Session,
    agent_id: str,
    query: str,
    transaction_id: uuid.UUID,
    include_memory: bool = True,
    streaming: bool = False, # Optional
    user_permissions: Optional[List[str]] = None,  # Optional user permissions for tool filtering
    media_items: Optional[List[Any]] = None,
) -> tuple[Any, Dict]:
    """
    Create a LangGraph agent workflow from database configuration.
    
    Args:
        db: Database session
        agent_id: ID of the agent to load
        query: The query for context (used in prompt preparation)
        transaction_id: Transaction ID for session tracking
        include_memory: Whether to include memory/chat history (default True)
        streaming: Whether to enable streaming (default False)
        user_permissions: Optional list of user permission strings for tool filtering
    Returns:
        tuple: (compiled_workflow, metadata_dict)
    """
    try:
        # 1. Get agent data using SAME generic database utils as CrewAI
        agent_data = AgentDatabaseManager.get_agent_data(db, agent_id)
        session_id = AgentDatabaseManager.get_session_from_transaction(db, transaction_id)
        config_sections = AgentDatabaseManager.extract_agent_config_sections(agent_data)
        
        # 2. Setup chat history from database interactions (same as CrewAI)
        chat_history = []
        if include_memory:
            from db.crud.interaction import get_agent_sessions_and_interactions
            paired_interactions = get_agent_sessions_and_interactions(db, agent_id, session_id)
            if len(paired_interactions["sessions"]) > 0:
                # Get only up to last 10 interactions
                chat_history = paired_interactions["sessions"][0]["interactions"][-10:]
        
        # 3. Initialize LLM using SAME generic factory as CrewAI
        llm = LLMFactory.create_llm(agent_data['config'])
        llm.streaming = streaming
        llm_metadata = LLMFactory.get_llm_metadata(agent_data['config'])
        
        # 4. Setup MCP tools using PROPER LangChain MCP tools manager
        langchain_mcp_manager = LangChainMCPToolsManager(
            config_sections['mcp_config'], 
            agent_config=agent_data['config'],
            user_permissions=user_permissions
        )
        langchain_mcp_tools = await langchain_mcp_manager.initialize_langchain_mcp_tools()
        
        # Get tools in the format needed for LangChain binding
        langchain_tool_defs = langchain_mcp_manager.get_tools_for_binding()
        
        logger.info(f"Initialized {len(langchain_mcp_tools)} native LangChain MCP tools")
        logger.info(f"Created {len(langchain_tool_defs)} tool definitions for binding")
        
        # 5. Prepare system prompt using SAME generic template manager as CrewAI
        base_prompt = TemplateManager.prepare_agent_prompt(
            agent_name=agent_data['name'],
            agent_description=agent_data['description'],
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            chat_history=chat_history,
            prompt_template=agent_data['prompt_template']
        )
        
        # 6. Get output format — only use it if explicitly set in DB or template file.
        # When absent, let the LLM respond naturally (same as CrewAI which passes a generic
        # expected_output but never injects extra system instructions around it).
        db_output_format = agent_data['output_format']
        has_explicit_output_format = db_output_format and db_output_format not in ["None", "", None]

        # Escape any curly braces in the prompt to prevent LangChain template variable conflicts
        base_prompt = base_prompt.replace('{', '{{').replace('}', '}}')
        
        # Build prompt messages: system prompt + optional output format + conversation
        messages = [("system", base_prompt)]
        if has_explicit_output_format:
            expected_output = TemplateManager.get_agent_output_format(
                agent_data['name'], db_output_format
            )
            escaped_expected_output = (expected_output or "").replace('{', '{{').replace('}', '}}')
            messages.append(("system", f"Expected output format:\n{escaped_expected_output}"))
        messages.append(MessagesPlaceholder(variable_name="messages"))
        prompt = ChatPromptTemplate.from_messages(messages)
        
        # Bind tools to LLM using converted tool definitions
        if langchain_tool_defs:
            try:
                agent_runnable = prompt | llm.bind_tools(langchain_tool_defs)
                logger.info(f"Successfully bound {len(langchain_tool_defs)} tools to LLM")
            except Exception as e:
                logger.error(f"Failed to bind tools: {e}")
                agent_runnable = prompt | llm
        else:
            agent_runnable = prompt | llm

        def agent_node(state: AgentState, config=None):
            """Agent node - SAME pattern as working langgraph_agent.py"""
            # Pass config to the agent runnable to enable callbacks
            if config:
                return {"messages": [agent_runnable.invoke(state, config=config)]}
            else:
                return {"messages": [agent_runnable.invoke(state)]}

        # Use standard ToolNode - SAME as working langgraph_agent.py
        tool_node = ToolNode(langchain_mcp_tools)

        def _should_continue(state: AgentState):
            """Decide whether to continue or end - SAME logic as working langgraph_agent.py"""
            if isinstance(state["messages"][-1], AIMessage) and not state["messages"][-1].tool_calls:
                return "end"
            return "continue"

        # 7. Build LangGraph workflow - SAME as working langgraph_agent.py
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", _should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        
        # 8. Compile workflow
        compiled_workflow = workflow.compile()
        
        # 9. Return workflow and metadata (similar to CrewAI pattern)
        metadata = {
            'agent_data': agent_data,
            'session_id': session_id,
            'chat_history': chat_history,
            'llm_metadata': llm_metadata,
            'tools': langchain_mcp_tools,
            'tool_defs': langchain_tool_defs,
            'mcp_manager': langchain_mcp_manager
        }
        
        return compiled_workflow, metadata
        
    except Exception as e:
        logger.error(f"Error creating LangGraph agent from config: {e}")
        raise


async def create_langgraph_agent_only(
    db: Session,
    agent_id: str,
    transaction_id: uuid.UUID,
    include_memory: bool = False
) -> tuple[Any, Dict]:
    """
    Create only a LangGraph agent workflow without execution.
    Convenience function for cases where no query is needed.
    
    Args:
        db: Database session
        agent_id: ID of the agent to load
        transaction_id: Transaction ID for session tracking
        include_memory: Whether to include memory/chat history (default False)
        
    Returns:
        tuple: (compiled_workflow, metadata_dict)
    """
    # Use a placeholder query for agent initialization
    placeholder_query = "Initialize agent"
    return await create_langgraph_agent_from_config(
        db, agent_id, placeholder_query, transaction_id, include_memory
    )


# LangGraph State Definition
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    input: str
    output: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]

# Using custom callback handler to log tool usage to the database
from langchain.callbacks.base import BaseCallbackHandler
class DBLogger(BaseCallbackHandler):
    def __init__(self, db: Session, transaction_id: uuid.UUID, agent_id: uuid.UUID, tool_usage: List[Dict], session_id: str):
        self.db = db
        self.transaction_id = transaction_id
        self.agent_id = agent_id
        self.tool_usage = tool_usage
        self.session_id = session_id
        self.tool_starts = {}  # Store tool start times and inputs by run_id
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        run_id = kwargs.get("run_id")
        start_time = datetime.now()
        
        # Store tool information for when it ends
        self.tool_starts[run_id] = {
            "tool_name": tool_name,
            "input_str": input_str,
            "start_time": start_time
        }
        
        logger.info(f"Tool started: {tool_name} | Input: {input_str[:100]}...")

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        end_time = datetime.now()
        
        if run_id in self.tool_starts:
            tool_info = self.tool_starts.pop(run_id)
            
            # Log tool execution to database using the modified function
            log_agent_step_langgraph(
                tool_name=tool_info["tool_name"],
                tool_input=tool_info["input_str"],
                tool_output=str(output),
                start_time=tool_info["start_time"],
                end_time=end_time,
                db=self.db,
                transaction_id=self.transaction_id,
                agent_id=self.agent_id,
                tool_usage=self.tool_usage,
                session_id=self.session_id
            )
            
            logger.info(f"Tool finished: {tool_info['tool_name']} | Output: {str(output)[:100]}...")
        else:
            logger.warning(f"Tool end event without corresponding start event for run_id: {run_id}")

async def get_and_run_langgraph_agent_v2(
    db: Session,
    agent_id: str,
    query: str,
    transaction_id: uuid.UUID,
    user_permissions: Optional[List[str]] = None,
    media_items: Optional[List[Any]] = None
) -> Dict:
    """
    LangGraph agent using generic modules for maximum code reuse.
    
    Args:
        db: Database session
        agent_id: ID of the agent to run
        query: User query
        transaction_id: Transaction ID for tracking
        user_permissions: Optional list of user permission strings for tool filtering
    """
    try:
        # 1. Create LangGraph agent using extracted function (same pattern as CrewAI)
        app, metadata = await create_langgraph_agent_from_config(
            db, agent_id, query, transaction_id, include_memory=True, user_permissions=user_permissions
        )
        
        # Extract metadata for execution
        session_id = metadata['session_id']
        chat_history = metadata['chat_history']
        llm_metadata = metadata['llm_metadata']
        langchain_mcp_tools = metadata['tools']
        langchain_mcp_manager = metadata['mcp_manager']
        agent_data = metadata['agent_data']
        
        # 2. Initialize tracking variables
        tool_usage = []
        start_time = datetime.now()
        
        # 9. Execute with SAME OpenTelemetry tracing as CrewAI
        span_attributes = create_agent_execution_span(
            span_name="langgraph-agent-kickoff",
            agent_name=agent_data['name'],
            agent_id=agent_id,
            session_id=session_id,
            query=query,
            tools=langchain_mcp_tools
        )
        
        with tracer.start_as_current_span("langgraph-agent-kickoff") as span:
            # Add SAME standardized span attributes as CrewAI
            for key, value in span_attributes.items():
                span.set_attribute(key, value)
            
            # Add LangGraph-specific attributes
            span.set_attribute("langgraph.nodes_count", 2)  # agent + tools nodes
            span.set_attribute("langgraph.has_tools", len(langchain_mcp_tools) > 0)
            span.set_attribute("langgraph.tool_definitions_count", len(metadata['tool_defs']))
            span.set_attribute("langgraph.native_langchain_tools", True)
            
            logger.info(f"Starting LangGraph workflow with span ID: {hex(span.get_span_context().span_id)}")
            
            # Execute the workflow - SAME as working langgraph_agent.py
            start_time_execution = datetime.now()
            
            # Convert database chat history to LangChain messages (include attachment metadata)
            langchain_messages = []
            from utils.media_storage import get_media_storage
            from agents.shared.message_builder import build_content as build_multimodal_content, media_items_from_attachment_metadata
            storage = get_media_storage()
            for interaction in chat_history:
                user_text = interaction.get("user_message") or interaction.get("user")
                if user_text:
                    user_attachments = interaction.get("user_attachments") or []
                    if user_attachments and storage:
                        history_media = media_items_from_attachment_metadata(storage, user_attachments)
                        content = build_multimodal_content(user_text, history_media)
                    else:
                        content = user_text
                    langchain_messages.append(HumanMessage(content=content))
                if interaction.get("agent_response") or interaction.get("assistant"):
                    assistant_text = interaction.get("agent_response") or interaction.get("assistant")
                    langchain_messages.append(AIMessage(content=assistant_text))
            
            # Add the current user query (and optional multi-modal content) to the messages
            content = build_multimodal_content(query, media_items)
            langchain_messages.append(HumanMessage(content=content))
            
            try:
                final_state = await app.ainvoke({"messages": langchain_messages}, config={"callbacks": [DBLogger(db, transaction_id, uuid.UUID(agent_id), tool_usage, session_id)]})
                execution_status = "success"
                final_output = final_state["messages"][-1].content
            except Exception as e:
                execution_status = "error"
                final_output = f"Error: {str(e)}"
                span.record_exception(e)
                logger.error(f"LangGraph execution failed: {e}")
                raise
            finally:
                end_time_execution = datetime.now()
                execution_time_ms = int((end_time_execution - start_time_execution).total_seconds() * 1000)
                
                # Add completion attributes using SAME pattern as CrewAI
                span.set_attribute("langgraph.execution_time_ms", execution_time_ms)
                span.set_attribute("langgraph.result_length", len(str(final_output)))
                span.set_attribute("langgraph.result_type", type(final_output).__name__)
                span.set_attribute("langgraph.execution_status", execution_status)
                span.set_attribute("gen_ai.completion", f"Final Answer: {final_output}")
        
        # 10. Log interactions to database (same as CrewAI)
        attachments = (
            [{"media_id": m.media_id, "media_type": m.media_type, "cloud_path": m.cloud_path} for m in (media_items or [])]
        ) if media_items else None
        if attachments is not None:
            attachments = list(attachments)
        from utils.user_context import get_user_context_from_transaction
        try:
            user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
            log_agent_interactions(
                db, session_id, uuid.UUID(agent_id), query, str(final_output),
                transaction_id=transaction_id, user_id=user_id, entity_id=entity_id,
                attachments=attachments,
            )
        except ValueError:
            log_agent_interactions(
                db, session_id, uuid.UUID(agent_id), query, str(final_output),
                transaction_id=transaction_id, attachments=attachments,
            )
        
        # 11. Use SAME LLM usage logging as CrewAI
        end_time = datetime.now()
        log_llm_usage(
            db=db,
            transaction_id=transaction_id,
            agent_id=uuid.UUID(agent_id),
            llm_metadata=llm_metadata,
            query=query,
            result=str(final_output),
            start_time=start_time,
            end_time=end_time
        )
        
        # 12. Use SAME response preparation as CrewAI
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        serialized_tool_usage = serialize_tool_usage(tool_usage)
        
        return {
            "result": str(final_output),
            "tool_usage": serialized_tool_usage,
            "llm_usage": {
                **llm_metadata,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": duration_ms
            },
            "total_duration_ms": duration_ms,
            "total_tools_used": len(serialized_tool_usage),
            "chat_history": chat_history
        }
        
    except Exception as e:
        logger.error(f"Error in LangGraph agent v2: {str(e)}")
        raise
    finally:
        # 13. Cleanup LangChain MCP resources
        if 'langchain_mcp_manager' in locals():
            await langchain_mcp_manager.cleanup()



async def execute_langgraph_workflow(
    db: Session,
    workflow_data: Dict,
    sub_agents_config: List[Dict],
    query: str,
    transaction_id: uuid.UUID,
    session_id: str,
    media_items: Optional[List[Any]] = None,
) -> str:
    """Execute workflow using LangGraph framework"""
    try:
        logger.info(f"🚀 Starting LangGraph Workflow")
        logger.info(f"📋 Query: {query}")
        logger.info(f"👥 Sub-agents: {[config['name'] for config in sub_agents_config]}")
        
        # Step 1: Create sub-nodes along with tools for each sub-agent (no tasks)
        sub_agent_workflows = {}
        sub_agent_metadata = {}
        
        for sub_agent_config in sub_agents_config:
            sub_agent_id = sub_agent_config['id']
            sub_agent_name = sub_agent_config['name']
            
            try:
                # Reuse create_langgraph_agent_only (no tasks, just agent setup)
                sub_workflow, sub_metadata = await create_langgraph_agent_only(
                    db, sub_agent_id, transaction_id, include_memory=False
                )
                
                sub_agent_workflows[sub_agent_id] = sub_workflow
                sub_agent_metadata[sub_agent_id] = sub_metadata
                
                logger.info(f"📦 Loaded: {sub_agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to create sub-node {sub_agent_id}: {e}")
                continue
        
        if not sub_agent_workflows:
            raise Exception("No valid sub-agent nodes could be created")
        
        # Step 2: Create workflow node along with its tools
        supervisor_workflow, supervisor_metadata = await create_langgraph_agent_from_config(
            db, workflow_data['id'], query, transaction_id, include_memory=True
        )
        
        logger.info("🎯 Loaded: Workflow Supervisor")
        
        # Step 3: Define the workflow graph structure using supervisor multi-agent architecture
        from typing import Literal
        from langgraph.types import Command
        from langgraph.graph import StateGraph, MessagesState, START, END
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Create supervisor node that routes to sub-agents
        async def supervisor(state: MessagesState) -> Command[Literal[*[f"agent_{config['id']}" for config in sub_agents_config], END]]:
            """Supervisor node that routes to sub-agents using database-configured prompts"""
            
            # Build available agents context
            available_agents = []
            for agent_config in sub_agents_config:
                agent_data = AgentDatabaseManager.get_agent_data(db, agent_config['id'])
                available_agents.append(f"- {agent_config['id']}: {agent_data['name']} - {agent_data['description']}")
            
            # Get supervisor's prompt template from database
            supervisor_data = AgentDatabaseManager.get_agent_data(db, workflow_data['id'])
            supervisor_prompt = supervisor_data.get('prompt_template', '')
            
            # Get conversation history from database interactions
            from db.crud.interaction import get_agent_sessions_and_interactions
            paired_interactions = get_agent_sessions_and_interactions(db, workflow_data['id'], session_id)
            db_conversation_history = []
            if len(paired_interactions["sessions"]) > 0:
                db_conversation_history = paired_interactions["sessions"][0]["interactions"][-10:]
            
            # Combine database history with current state messages
            conversation_history = [msg.content for msg in state["messages"]]
            if db_conversation_history:
                db_context = "Previous workflow interactions:\n"
                for interaction in db_conversation_history:
                    user_text = interaction.get("user_message") or interaction.get("user")
                    if user_text:
                        att = interaction.get("user_attachments") or []
                        if att:
                            db_context += f"User: {user_text} [Attached: {len(att)} file(s)]\n"
                        else:
                            db_context += f"User: {user_text}\n"
                    if interaction.get("agent_response") or interaction.get("assistant"):
                        db_context += f"Agent: {interaction.get('agent_response') or interaction.get('assistant')}\n"
                conversation_history = [db_context] + conversation_history
            
            # Count how many messages we have and check for AI responses (indicating sub-agent work)
            num_messages = len(state["messages"])
            has_ai_responses = any(isinstance(msg, AIMessage) for msg in state["messages"])
            
            has_sub_agent_responses = has_ai_responses and num_messages > 1
            
            # Log workflow progression
            if has_sub_agent_responses:
                logger.info("🔄 Workflow in progress - supervisor deciding next step")
            else:
                logger.info("🚀 Starting workflow - supervisor delegating first task")
            
            # If we have sub-agent responses, check if workflow is complete
            if has_sub_agent_responses and len(state["messages"]) > 2:
                # Track executed agents using metadata stored in AIMessages
                executed_agent_ids = []
                executed_agent_names = []
                
                # Check each AIMessage for agent metadata
                for msg in state["messages"]:
                    if isinstance(msg, AIMessage):
                        # Check if this message has agent metadata (from sub-agent execution)
                        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                            agent_id = msg.additional_kwargs.get('agent_id')
                            agent_name = msg.additional_kwargs.get('agent_name')
                            if agent_id and agent_id not in executed_agent_ids:
                                executed_agent_ids.append(agent_id)
                                executed_agent_names.append(agent_name or agent_id)
                
                # Get remaining agents that haven't been executed
                remaining_agents = [
                    f"- {config['id']}: {AgentDatabaseManager.get_agent_data(db, config['id'])['name']} - {AgentDatabaseManager.get_agent_data(db, config['id'])['description']}"
                    for config in sub_agents_config
                    if config['id'] not in executed_agent_ids
                ]
                
                # Get remaining agent IDs for routing logic
                remaining_agent_ids = [config['id'] for config in sub_agents_config if config['id'] not in executed_agent_ids]
                
                logger.info(f"📋 Agents executed so far: {executed_agent_names} (IDs: {executed_agent_ids})")
                logger.info(f"🔄 Remaining agents: {[AgentDatabaseManager.get_agent_data(db, agent_id)['name'] for agent_id in remaining_agent_ids]}")
                
                if not remaining_agents:
                    # All agents have been executed, workflow should complete
                    logger.info("✅ All agents executed, preparing final result")
                    routing_context = f"""You are a workflow supervisor. All sub-agents have completed their tasks.

                    Original query: {query}

                    Agents executed:
                    {chr(10).join([f"- {name}" for name in executed_agent_names])}

                    Your workflow responsibilities:
                    {supervisor_prompt}

                    Based on the work completed by all agents above, provide the final, synthesized result to the user. Respond with "finished" to complete the workflow."""
                else:
                    routing_context = f"""You are a workflow supervisor. Review the conversation and decide the next step:

                    Original query: {query}

                    Agents already executed:
                    {chr(10).join([f"- {name}" for name in executed_agent_names]) if executed_agent_names else "None"}

                    Remaining agents to execute:
                    {chr(10).join(remaining_agents)}

                    Your workflow responsibilities:
                    {supervisor_prompt}

                    IMPORTANT: Do NOT call agents that have already been executed. Call the next agent in sequence from the remaining agents list above.
                    Specify the exact agent_id from the remaining agents. If all work is complete, respond with "finished"."""
            else:
                # Initial routing - start the workflow
                routing_context = f"""You are a workflow supervisor. Start the workflow by delegating to the appropriate agent.

                    Available sub-agents:
                    {chr(10).join(available_agents)}

                    Current query: {query}

                    Your workflow responsibilities:
                    {supervisor_prompt}

                    Based on your workflow responsibilities above, determine which agent should start the workflow. Include the agent_id in your response."""
            
            # Use the pre-built supervisor workflow with full context
            supervisor_state = {"messages": state["messages"] + [HumanMessage(content=routing_context)]}
            supervisor_result = await supervisor_workflow.ainvoke(supervisor_state)
            
            # Parse supervisor's decision for routing
            supervisor_response = supervisor_result["messages"][-1].content
            decision = supervisor_response.strip().lower()
            
            # Log the supervisor's decision for debugging
            logger.info(f"🤔 Supervisor decision: {decision[:200]}...")
            
            # Check for completion signals - look for workflow completion
            completion_signals = ["finished", "complete", "done", "__end__", "final result", "workflow complete", "ready to send"]
            if any(signal in decision for signal in completion_signals):
                logger.info("✅ Workflow completed by supervisor")
                return Command(goto=END)
            
            # Check if decision contains any agent ID
            # First, check if we have executed agents (to prevent re-routing)
            executed_agent_ids_for_check = []
            if has_sub_agent_responses:
                for msg in state["messages"]:
                    if isinstance(msg, AIMessage) and hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                        agent_id = msg.additional_kwargs.get('agent_id')
                        if agent_id:
                            executed_agent_ids_for_check.append(agent_id.lower())
            
            # Try to find a valid agent to route to
            routed_agent = None
            for agent_config in sub_agents_config:
                if agent_config['id'].lower() in decision:
                    # Prevent re-routing to already executed agents
                    if agent_config['id'].lower() in executed_agent_ids_for_check:
                        logger.warning(f"⚠️  Supervisor tried to re-route to already executed agent: {agent_config['name']}. Ignoring.")
                        continue
                    
                    routed_agent = agent_config
                    agent_name = agent_config['name']
                    logger.info(f"➡️  Delegating to: {agent_name} ({agent_config['id']})")
                    return Command(goto=f"agent_{agent_config['id']}")
            
            # If no valid routing was found but we have remaining agents, route to the first remaining one
            if has_sub_agent_responses and not routed_agent:
                # Get remaining agents (executed_agent_ids_for_check is already populated above)
                remaining_agent_ids_for_fallback = []
                for config in sub_agents_config:
                    if config['id'].lower() not in executed_agent_ids_for_check:
                        remaining_agent_ids_for_fallback.append(config)
                
                if remaining_agent_ids_for_fallback:
                    # Route to the first remaining agent
                    fallback_agent = remaining_agent_ids_for_fallback[0]
                    logger.info(f"🔄 No clear routing in supervisor response, falling back to first remaining agent: {fallback_agent['name']}")
                    return Command(goto=f"agent_{fallback_agent['id']}")
            
            # If we've already had sub-agent responses and no clear routing, end the workflow
            if has_sub_agent_responses:
                logger.info("⚠️  No clear routing found, ending workflow")
                return Command(goto=END)
            
            # Default: end workflow if no clear routing found
            logger.warning(f"❌ No clear routing found in supervisor response: {supervisor_response}")
            return Command(goto=END)
        
        # Create sub-agent nodes that return to supervisor
        def create_sub_agent_node(agent_id: str):
            async def agent_node(state: MessagesState) -> Command[Literal["supervisor"]]:
                """Sub-agent node that executes and returns to supervisor"""
                # Get agent name for logging
                agent_data = AgentDatabaseManager.get_agent_data(db, agent_id)
                agent_name = agent_data['name']
                logger.info(f"🔧 Executing: {agent_name}")
                
                try:
                    # Use the pre-loaded workflow instead of reinitializing
                    if agent_id not in sub_agent_workflows:
                        raise Exception(f"Agent {agent_id} not found in pre-loaded workflows")
                    
                    # Get the pre-loaded workflow
                    agent_workflow = sub_agent_workflows[agent_id]
                    
                    # Execute with current context
                    agent_context = f"Workflow query: {query}\n\nProvide your specialized response."
                    
                    # Convert current state to LangChain messages
                    langchain_messages = []
                    for msg in state["messages"]:
                        if isinstance(msg, HumanMessage):
                            langchain_messages.append(msg)
                        elif isinstance(msg, AIMessage):
                            langchain_messages.append(msg)
                    
                    # Add the agent context
                    langchain_messages.append(HumanMessage(content=agent_context))
                    
                    # Execute the pre-loaded workflow
                    result = await agent_workflow.ainvoke({"messages": langchain_messages})
                    
                    # Extract the result
                    if result and "messages" in result and result["messages"]:
                        result_content = result["messages"][-1].content
                    else:
                        result_content = "No response generated"
                    
                    # Add agent identification to the message for better tracking
                    agent_response = AIMessage(
                        content=result_content,
                        additional_kwargs={"agent_id": agent_id, "agent_name": agent_name}
                    )
                    
                    logger.info(f"✅ Completed: {agent_name}")
                    
                    # Return to supervisor with APPENDED message (not replacement)
                    return Command(
                        goto="supervisor",
                        update={"messages": state["messages"] + [agent_response]}
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Failed: {agent_name} - {e}")
                    error_response = AIMessage(content=f"Agent {agent_name} error: {str(e)}")
                    
                    return Command(
                        goto="supervisor",
                        update={"messages": state["messages"] + [error_response]}
                    )
            
            return agent_node
        
        # Step 4: Build supervisor multi-agent architecture (https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
        builder = StateGraph(MessagesState)
        
        # Add supervisor node
        builder.add_node("supervisor", supervisor)
        
        # Add sub-agent nodes
        for agent_config in sub_agents_config:
            node_name = f"agent_{agent_config['id']}"
            builder.add_node(node_name, create_sub_agent_node(agent_config['id']))
        
        # Define workflow graph structure - supervisor architecture
        builder.add_edge(START, "supervisor")
        
        # Compile the workflow graph
        workflow_graph = builder.compile()
        
        # Step 5: Execute the workflow
        start_time = datetime.now()
        
        span_attributes = create_agent_execution_span(
            span_name="langgraph-supervisor-workflow",
            agent_name=workflow_data['name'],
            agent_id=workflow_data['id'],
            session_id=session_id,
            query=query,
            tools=[]  # Each agent has its own tools
        )
        
        with tracer.start_as_current_span("langgraph-supervisor-workflow") as span:
            # Set span attributes
            for key, value in span_attributes.items():
                span.set_attribute(key, value)
            
            # Execute the workflow with initial state and recursion limit
            # Increase limit to accommodate multiple agents (limit = number of agents * 2 + supervisor iterations)
            recursion_limit = len(sub_agents_config) * 3 + 5  # Allow 3 iterations per agent + 5 for supervisor
            initial_state = MessagesState(messages=[HumanMessage(content=query)])
            config = {"recursion_limit": recursion_limit}
            logger.info(f"🔧 Set recursion limit to {recursion_limit} for {len(sub_agents_config)} sub-agents")
            final_state = await workflow_graph.ainvoke(initial_state, config=config)
            
            # Extract final result from messages
            final_result = final_state['messages'][-1].content if final_state['messages'] else "Workflow completed"
            
            logger.info("🎉 Workflow Execution Complete")
            logger.info(f"📊 Total messages: {len(final_state['messages'])}")
            logger.info(f"⏱️  Duration: {int((datetime.now() - start_time).total_seconds())}s")
            
            # Log execution (reuse existing logging functions)
            end_time = datetime.now()
            llm_metadata = LLMFactory.get_llm_metadata(workflow_data['config'])
            log_llm_usage(db, transaction_id, uuid.UUID(workflow_data['id']), 
                         llm_metadata, query, str(final_result), start_time, end_time)
            # Get user context from transaction for interactions
            from utils.user_context import get_user_context_from_transaction
            try:
                user_id, entity_id = get_user_context_from_transaction(db, transaction_id)
                log_agent_interactions(
                    db, session_id, uuid.UUID(workflow_data['id']), 
                    query, str(final_result),
                    transaction_id=transaction_id, user_id=user_id, entity_id=entity_id
                )
            except ValueError:
                # Transaction not found, will use defaults
                log_agent_interactions(
                    db, session_id, uuid.UUID(workflow_data['id']), 
                    query, str(final_result),
                    transaction_id=transaction_id
                )
            
            span.set_attribute("workflow.sub_agents_count", len(sub_agents_config))
            span.set_attribute("workflow.framework", "langgraph_supervisor")
            span.set_attribute("workflow.messages_count", len(final_state['messages']))
        
        # Cleanup (reuse existing cleanup pattern)
        for sub_metadata in sub_agent_metadata.values():
            if sub_metadata.get('mcp_manager'):
                await sub_metadata['mcp_manager'].cleanup()
        if supervisor_metadata.get('mcp_manager'):
            await supervisor_metadata['mcp_manager'].cleanup()
            
        return str(final_result)
        
    except Exception as e:
        logger.error(f"Error in langgraph workflow execution: {e}")
        raise
