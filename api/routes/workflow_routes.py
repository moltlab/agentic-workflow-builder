import os
import base64
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid
import json
from datetime import datetime

from db.config import SessionLocal
from db.crud.agent import create_agent, get_agent_by_id, list_agents, AgentCreate
from utils.logging_utils import get_logger

# Import agent execution functions
from agents.langgraph_agent_v2 import get_and_run_langgraph_agent_v2, execute_langgraph_workflow
from agents.hierarchical_workflow_builder import HierarchicalWorkflowBuilder
from agents.config_driven_workflow import ConfigDrivenWorkflowBuilder

# Import transaction logging
from api.routes.agent_routes import log_transaction
from db.models import Transaction

# Import shared modules for workflow execution
from agents.shared import AgentDatabaseManager
from agents.shared.workflow_utils import build_input_messages, build_node_agent_map
from agents.shared.agent_logging import log_agent_interactions, log_agent_step_langgraph, log_llm_usage
from db.crud.agent_execution_log import create_agent_execution_log
from langchain_core.messages import HumanMessage, AIMessage

logger = get_logger('workflow_routes')

# File-based workflow defaults (when DB / request body does not supply config)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKFLOW_DEFAULTS_DIR = _PROJECT_ROOT / "config" / "workflow_defaults"

workflow_router = APIRouter()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for workflow management
class WorkflowAgent(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class WorkflowQueryRequest(BaseModel):
    workflow_id: str
    query: str
    framework: Optional[str] = "langgraph"
    session_id: Optional[str] = None

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agents: List[WorkflowAgent]
    llm_used: str = "openai"  # LLM provider
    llm_config: Dict[str, Any] = {}  # LLM configuration
    prompt_template: Optional[str] = None
    output_format: Optional[str] = None
    framework: Optional[str] = "langgraph"

class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow with optional fields"""
    name: Optional[str] = None
    description: Optional[str] = None
    agents: Optional[List[WorkflowAgent]] = None
    llm_used: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None
    output_format: Optional[str] = None
    framework: Optional[str] = None

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    agents: List[WorkflowAgent]
    config: Dict[str, Any]
    llm_used: str
    prompt_template: Optional[str]
    output_format: Optional[str]
    status: str
    type: Optional[str]

def load_workflow_and_agents_from_db(db: Session, workflow_id: str):
    """
    Load workflow configuration and referenced agents directly from the database.
    Returns (workflow_config: dict, agents_config: dict).
    Raises HTTPException on validation or lookup errors.
    """
    try:
        workflow_agent = get_agent_by_id(db, workflow_id)
        if not workflow_agent:
            raise HTTPException(status_code=404, detail="Workflow config not found")
        # Accept both hierarchical_workflow and supervisor_workflow types for loading
        valid_workflow_agent_types = ["hierarchical_workflow", "supervisor_workflow", "workflow"]
        if getattr(workflow_agent, "type", None) not in valid_workflow_agent_types:
            raise HTTPException(status_code=400, detail="Agent is not a workflow agent")

        workflow_config = (workflow_agent.config or {}).get("workflow")
        if not workflow_config:
            raise HTTPException(status_code=400, detail="Workflow config missing in agent record")

        nodes = workflow_config.get("nodes") or []
        if not isinstance(nodes, list):
            raise HTTPException(status_code=400, detail="Invalid workflow config: nodes must be a list")

        agents_config = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            agent_id = node.get("agent")
            if not agent_id:
                continue
            agent_obj = get_agent_by_id(db, agent_id)
            if not agent_obj:
                raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
            agent_conf = agent_obj.config or {}
            agent_conf = dict(agent_conf)

            valid_workflow_types = ["simple", "llm", "llm_with_tools"]

            if "type" not in agent_conf:
                mcp_tools = agent_conf.get("mcp", {}).get("tools", [])
                direct_tools = agent_conf.get("tools", [])
                tools_list = mcp_tools or direct_tools
                agent_type = getattr(agent_obj, "type", None)
                if agent_type and agent_type in valid_workflow_types:
                    agent_conf["type"] = agent_type
                else:
                    agent_conf["type"] = "llm_with_tools" if tools_list else "llm"
            else:
                config_type = agent_conf.get("type")
                if config_type not in valid_workflow_types:
                    mcp_tools = agent_conf.get("mcp", {}).get("tools", [])
                    direct_tools = agent_conf.get("tools", [])
                    tools_list = mcp_tools or direct_tools
                    agent_conf["type"] = "llm_with_tools" if tools_list else "llm"
                else:
                    agent_conf["type"] = config_type

            if agent_conf.get("type") in ["llm", "llm_with_tools"]:
                if "prompt" not in agent_conf:
                    if getattr(agent_obj, "prompt_template", None):
                        agent_conf["prompt"] = agent_obj.prompt_template
                    elif agent_conf.get("prompt_template"):
                        agent_conf["prompt"] = agent_conf.get("prompt_template")
                    else:
                        agent_conf["prompt"] = ""

            if "output_fields" not in agent_conf:
                output_format = getattr(agent_obj, "output_format", None)
                if output_format:
                    try:
                        if isinstance(output_format, str):
                            output_fields = json.loads(output_format)
                        else:
                            output_fields = output_format
                        if isinstance(output_fields, list):
                            agent_conf["output_fields"] = output_fields
                        else:
                            agent_conf["output_fields"] = [output_fields]
                    except (json.JSONDecodeError, TypeError):
                        agent_conf["output_fields"] = ["messages"]
                else:
                    agent_conf["output_fields"] = ["messages"]

            if agent_conf.get("type") == "llm_with_tools":
                mcp_tools = agent_conf.get("mcp", {}).get("tools", [])
                if mcp_tools:
                    agent_conf["tools"] = mcp_tools

            agents_config[str(agent_obj.id)] = agent_conf

        if "workflow_name" not in workflow_config:
            raise HTTPException(status_code=400, detail="Workflow config missing required field: workflow_name")
        if "edges" not in workflow_config or not isinstance(workflow_config["edges"], list):
            raise HTTPException(status_code=400, detail="Workflow config missing or invalid edges")

        return workflow_config, agents_config
    except HTTPException:
        raise
    except Exception:
        raise

def update_workflow_config_selectively(existing_config: Dict[str, Any], workflow_data: WorkflowUpdate) -> Dict[str, Any]:
    """
    Update workflow configuration selectively, only modifying the fields that are provided.
    This preserves existing configuration values that are not being updated.
    """
    updated_config = existing_config.copy()
    
    # Initialize workflow section if it doesn't exist
    if 'workflow' not in updated_config:
        updated_config['workflow'] = {}
    
    if workflow_data.agents is not None:
        updated_config['workflow']['agents'] = [
            {
                'id': agent.id,
                'name': agent.name
            }
            for agent in workflow_data.agents
        ]
    
    # Update LLM configuration selectively
    if workflow_data.llm_used is not None or workflow_data.llm_config is not None:
        if 'llms' not in updated_config:
            updated_config['llms'] = {}
        
        # Get the LLM provider (use existing if not provided)
        llm_provider = workflow_data.llm_used or existing_config.get('llm_used', 'openai')
        
        # Initialize LLM config if it doesn't exist
        if llm_provider not in updated_config['llms']:
            updated_config['llms'][llm_provider] = {}
        
        # Update LLM config only if provided
        if workflow_data.llm_config is not None:
            updated_config['llms'][llm_provider].update(workflow_data.llm_config)
    
    return updated_config

@workflow_router.get("/workflows", response_model=List[WorkflowResponse])
async def get_workflows(
    db: Session = Depends(get_db)):
    """Get all workflow agents"""
    try:
        # Query for agents that have workflow configuration
        agents = list_agents(db)
        workflows = []
        
        for agent in agents:
            # Check if this is a supervisor workflow agent by type
            if hasattr(agent, 'type') and agent.type == 'supervisor_workflow':
                # This is a supervisor workflow agent
                workflow_agents = []
                
                # Extract workflow agents from config and fetch full agent data
                if agent.config and isinstance(agent.config, dict):
                    workflow_config = agent.config.get('workflow', {})
                    for agent_ref in workflow_config.get('agents', []):
                        agent_id = agent_ref.get('id', '')
                        agent_name = agent_ref.get('name', '')
                        
                        # Fetch full agent data from database
                        try:
                            full_agent = get_agent_by_id(db, agent_id)
                            if full_agent:
                                workflow_agents.append(WorkflowAgent(
                                    id=agent_id,
                                    name=agent_name,
                                    description=full_agent.description,
                                    config=full_agent.config
                                ))
                            else:
                                # Fallback if agent not found
                                workflow_agents.append(WorkflowAgent(
                                    id=agent_id,
                                    name=agent_name,
                                    description="Agent not found",
                                    config={}
                                ))
                        except Exception as e:
                            logger.error(f"Error fetching agent {agent_id}: {e}")
                            # Fallback if error occurs
                            workflow_agents.append(WorkflowAgent(
                                id=agent_id,
                                name=agent_name,
                                description="Error loading agent",
                                config={}
                            ))
                
                workflows.append(WorkflowResponse(
                    id=str(agent.id),
                    name=agent.name,
                    description=agent.description,
                    agents=workflow_agents,
                    status='active' if agent.is_active else 'inactive',
                    config=agent.config,
                    llm_used=agent.llm_used,
                    prompt_template=agent.prompt_template,
                    output_format=agent.output_format,
                    type=agent.type
                ))
        
        return workflows
    except Exception as e:
        logger.error(f"Error fetching workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: Session = Depends(get_db)):
    """Get a specific workflow by ID"""
    try:
        agent = get_agent_by_id(db, workflow_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_config = agent.config.get('workflow', {})
        if not workflow_config:
            raise HTTPException(status_code=404, detail="Agent is not a workflow")
        
        workflow_agents = []
        for agent_ref in workflow_config.get('agents', []):
            workflow_agents.append(WorkflowAgent(
                id=agent_ref.get('id', ''),
                name=agent_ref.get('name', '')
            ))
        
        return WorkflowResponse(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            agents=workflow_agents,
            status='active' if agent.is_active else 'inactive',
            config=agent.config,
            llm_used=agent.llm_used,
            prompt_template=agent.prompt_template,
            output_format=agent.output_format,
            type=agent.type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: Session = Depends(get_db)):
    """Create a new workflow agent"""
    try:
        # Validate that all referenced agents exist
        for agent_ref in workflow_data.agents:
            existing_agent = get_agent_by_id(db, agent_ref.id)
            if not existing_agent:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Agent with ID {agent_ref.id} not found"
                )
        
        # Create workflow configuration with agent references
        workflow_config = {
            'workflow': {
                'agents': [{'id': agent.id, 'name': agent.name} for agent in workflow_data.agents]
            },
            'mcp': {'enabled': False, 'server_url': 'http://localhost:8001', 'tools': []},
            'llms': {workflow_data.llm_used: workflow_data.llm_config} if workflow_data.llm_config else {}
        }
        
        # Create supervisor workflow agent using existing create_agent function - same as agent_routes.py
        agent_create = AgentCreate(
            name=workflow_data.name,
            description=workflow_data.description or f"Workflow with {len(workflow_data.agents)} agents",
            type="supervisor_workflow",
            config=workflow_config,
            prompt_template=workflow_data.prompt_template or "You are a workflow coordinator that manages multiple specialized agents.",
            output_format=workflow_data.output_format or "Provide a comprehensive response based on the coordinated work of all agents in the workflow.",
            llm_used=workflow_data.llm_used
        )
        
        created_agent = create_agent(db, agent_create)
        
        return WorkflowResponse(
            id=str(created_agent.id),
            name=created_agent.name,
            description=created_agent.description,
            agents=workflow_data.agents,
            status='active' if created_agent.is_active else 'inactive',
            config=created_agent.config,
            llm_used=created_agent.llm_used,
            prompt_template=created_agent.prompt_template,
            output_format=created_agent.output_format,
            type=created_agent.type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    db: Session = Depends(get_db)):
    """Delete a workflow agent"""
    try:
        from db.crud.agent import delete_agent
        
        # Verify it's a workflow agent
        agent = get_agent_by_id(db, workflow_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_config = agent.config.get('workflow')
        if not workflow_config:
            raise HTTPException(status_code=400, detail="Agent is not a workflow")
        
        # Delete the workflow agent
        delete_agent(db, workflow_id)
        
        return {"message": "Workflow deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str, 
    workflow_data: WorkflowCreate, 
    db: Session = Depends(get_db)):
    """Update an existing workflow"""
    try:
        from db.crud.agent import update_agent, AgentUpdate
        
        # Get existing workflow
        agent = get_agent_by_id(db, workflow_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_config = agent.config.get('workflow')
        if not workflow_config:
            raise HTTPException(status_code=400, detail="Agent is not a workflow")
        
        # Validate that all referenced agents exist
        for agent_ref in workflow_data.agents:
            existing_agent = get_agent_by_id(db, agent_ref.id)
            if not existing_agent:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Agent with ID {agent_ref.id} not found"
                )
        
        # Update workflow configuration
        updated_config = agent.config.copy()
        updated_config['workflow'] = {
            'framework': workflow_data.framework,
            'agents': [
                {
                    'id': agent.id,
                    'name': agent.name
                }
                for agent in workflow_data.agents
            ]
        }
        
        # Update the agent
        agent_update = AgentUpdate(
            name=workflow_data.name,
            description=workflow_data.description,
            config=updated_config
        )
        
        updated_agent = update_agent(db, workflow_id, agent_update)
        
        return WorkflowResponse(
            id=str(updated_agent.id),
            name=updated_agent.name,
            description=updated_agent.description,
            config=updated_agent.config,
            llm_used=updated_agent.llm_used,
            prompt_template=updated_agent.prompt_template,
            output_format=updated_agent.output_format,
            agents=workflow_data.agents,
            status='active' if updated_agent.is_active else 'inactive',
            type=updated_agent.type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.patch("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow_selective(
    workflow_id: str, 
    workflow_data: WorkflowUpdate, 
    db: Session = Depends(get_db)):
    """Update specific fields of an existing workflow (selective update)"""
    try:
        from db.crud.agent import update_agent, AgentUpdate
        
        # Get existing workflow
        agent = get_agent_by_id(db, workflow_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check for supervisor_workflow type
        if agent.type != 'supervisor_workflow':
            raise HTTPException(status_code=400, detail="Agent is not a supervisor workflow")
        
        # Validate that all referenced agents exist (only if agents are being updated)
        if workflow_data.agents is not None:
            for agent_ref in workflow_data.agents:
                # Validate UUID format first
                try:
                    import uuid
                    uuid.UUID(agent_ref.id)
                except ValueError:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid agent ID format: {agent_ref.id}"
                    )
                
                existing_agent = get_agent_by_id(db, agent_ref.id)
                if not existing_agent:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Agent with ID {agent_ref.id} not found"
                    )
        
        # Update workflow configuration selectively
        updated_config = update_workflow_config_selectively(agent.config, workflow_data)
        
        # Prepare agent update data (only include fields that are provided)
        agent_update_data = {}
        
        if workflow_data.name is not None:
            agent_update_data['name'] = workflow_data.name
        if workflow_data.description is not None:
            agent_update_data['description'] = workflow_data.description
        if workflow_data.llm_used is not None:
            agent_update_data['llm_used'] = workflow_data.llm_used
        if workflow_data.prompt_template is not None:
            agent_update_data['prompt_template'] = workflow_data.prompt_template
        if workflow_data.output_format is not None:
            agent_update_data['output_format'] = workflow_data.output_format
        
        # Always update config if any workflow-related fields were provided
        if any([
            workflow_data.framework is not None,
            workflow_data.agents is not None,
            workflow_data.llm_config is not None
        ]):
            agent_update_data['config'] = updated_config
        
        # If no fields are provided, return the existing agent without updating
        if not agent_update_data:
            updated_agent = agent
        else:
            # Update the agent
            agent_update = AgentUpdate(**agent_update_data)
            updated_agent = update_agent(db, workflow_id, agent_update)
        
        # Get agents for response (use provided agents or existing ones)
        response_agents = workflow_data.agents if workflow_data.agents is not None else [
            WorkflowAgent(
                id=agent_ref.get('id', ''),
                name=agent_ref.get('name', ''),
                description=None,
                config=None
            )
            for agent_ref in updated_agent.config.get('workflow', {}).get('agents', [])
        ]
        
        return WorkflowResponse(
            id=str(updated_agent.id),
            name=updated_agent.name,
            description=updated_agent.description,
            config=updated_agent.config,
            llm_used=updated_agent.llm_used,
            prompt_template=updated_agent.prompt_template,
            output_format=updated_agent.output_format,
            agents=response_agents,
            status='active' if updated_agent.is_active else 'inactive',
            type=updated_agent.type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow {workflow_id} selectively: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Workflow configuration and execution APIs (moved from web_interface.py)
# ------------------------------------------------------------

@workflow_router.get("/workflow/config")
async def get_workflow_config():
    """Get current workflow configuration"""
    try:
        config_path = str(_WORKFLOW_DEFAULTS_DIR / "workflow_config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Workflow config file not found")

        with open(config_path, 'r') as f:
            workflow_config = json.load(f)

        agents_config_path = str(_WORKFLOW_DEFAULTS_DIR / "agents_config.json")
        agents_config = {}
        if os.path.exists(agents_config_path):
            with open(agents_config_path, 'r') as f:
                agents_config = json.load(f)

        return JSONResponse({"workflow_config": workflow_config, "agents_config": agents_config})
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in config file: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading workflow config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading workflow config: {str(e)}")


@workflow_router.post("/workflow/update")
async def update_workflow_config(
    request: Request):
    """Update workflow configuration"""
    try:
        data = await request.json()

        if "workflow_name" not in data:
            raise HTTPException(status_code=400, detail="Missing required field: workflow_name")
        if "nodes" not in data or not isinstance(data["nodes"], list):
            raise HTTPException(status_code=400, detail="Missing or invalid field: nodes (must be an array)")
        if "edges" not in data or not isinstance(data["edges"], list):
            raise HTTPException(status_code=400, detail="Missing or invalid field: edges (must be an array)")

        config_path = str(_WORKFLOW_DEFAULTS_DIR / "workflow_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Workflow config updated: {data.get('workflow_name', 'unknown')}")
        return JSONResponse({"message": "Workflow configuration saved successfully", "workflow_name": data.get("workflow_name")})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving workflow config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving workflow config: {str(e)}")


# ------------------------------------------------------------
# Database-based Hierarchical Workflow Configuration APIs
# ------------------------------------------------------------

@workflow_router.get("/workflow/hierarchical-workflows")
async def list_hierarchical_workflows(
    db: Session = Depends(get_db)):
    """
    List all hierarchical workflow agents from database.
    Returns workflows that have type='hierarchical_workflow'.
    """
    try:
        agents = list_agents(db)
        hierarchical_workflows = []
        
        for agent in agents:
            # Check if this is a hierarchical workflow agent
            if getattr(agent, 'type', None) != 'hierarchical_workflow':
                continue
            
            # Get workflow config for display info
            config = agent.config or {}
            workflow_config = config.get('workflow', {})
            nodes = workflow_config.get('nodes', [])
            
            # Extract node summary for card display
            nodes_summary = []
            for node in nodes:
                node_info = {
                    "id": node.get('id', ''),
                    "name": node.get('name', node.get('id', 'Unnamed')),
                    "type": node.get('type', 'simple'),
                    "agent_id": node.get('agent_id', None)
                }
                nodes_summary.append(node_info)
            
            # Add to the list with config details
            hierarchical_workflows.append({
                "id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "workflow_name": workflow_config.get('workflow_name', agent.name),
                "nodes_count": len(nodes),
                "edges_count": len(workflow_config.get('edges', [])),
                "nodes": nodes_summary,
                "entry_point": workflow_config.get('entry_point', ''),
                "is_active": getattr(agent, 'is_active', True)
            })
        
        return JSONResponse({"workflows": hierarchical_workflows})
    except Exception as e:
        logger.error(f"Error listing hierarchical workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing hierarchical workflows: {str(e)}")


@workflow_router.post("/workflow/hierarchical")
async def create_hierarchical_workflow(
    request: Request,
    db: Session = Depends(get_db)):
    """
    Create a new hierarchical workflow.
    Expects JSON body with workflow configuration including nodes and edges.
    """
    try:
        data = await request.json()
        
        # Validate required fields
        workflow_name = data.get('workflow_name')
        if not workflow_name:
            raise HTTPException(status_code=400, detail="workflow_name is required")
        
        # Extract workflow configuration
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        state_schema = data.get('state_schema', {"query": "str", "messages": "list"})
        description = data.get('description', f"Hierarchical workflow: {workflow_name}")
        
        # Determine entry_point from nodes or use provided value
        entry_point = data.get('entry_point')
        if not entry_point:
            # Find entry node or use first node
            for node in nodes:
                if node.get('type') == 'entry':
                    entry_point = node.get('id')
                    break
            if not entry_point and nodes:
                entry_point = nodes[0].get('id')
        
        # Build workflow config structure
        workflow_config = {
            'workflow': {
                'workflow_name': workflow_name,
                'state_schema': state_schema,
                'nodes': nodes,
                'edges': edges,
                'entry_point': entry_point
            },
            'mcp': {'enabled': False, 'server_url': 'http://localhost:8001', 'tools': []},
            'llms': {}
        }
        
        # Create the hierarchical workflow agent
        agent_create = AgentCreate(
            name=workflow_name,
            description=description,
            type="hierarchical_workflow",
            config=workflow_config,
            prompt_template=data.get('prompt_template', "You are a hierarchical workflow coordinator."),
            output_format=data.get('output_format', "Provide a comprehensive response."),
            llm_used=data.get('llm_used', 'openai'),
            is_active=True
        )
        
        created_agent = create_agent(db, agent_create)
        
        return JSONResponse({
            "success": True,
            "message": f"Hierarchical workflow '{workflow_name}' created successfully",
            "workflow": {
                "id": str(created_agent.id),
                "name": created_agent.name,
                "description": created_agent.description,
                "type": created_agent.type,
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "entry_point": entry_point
            }
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating hierarchical workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating hierarchical workflow: {str(e)}")


@workflow_router.get("/workflow/hierarchical-config/{workflow_id}")
async def get_hierarchical_workflow_config(
    workflow_id: str,
    db: Session = Depends(get_db)):
    """
    Get hierarchical workflow configuration from database.
    Returns the workflow config (nodes, edges, state_schema) and referenced agents config.
    """
    try:
        workflow_config, agents_config = load_workflow_and_agents_from_db(db, workflow_id)
        
        # Get workflow agent details
        workflow_agent = get_agent_by_id(db, workflow_id)
        
        return JSONResponse({
            "workflow_id": workflow_id,
            "workflow_name": workflow_agent.name if workflow_agent else workflow_config.get('workflow_name', 'Unknown'),
            "workflow_description": workflow_agent.description if workflow_agent else '',
            "workflow_config": workflow_config,
            "agents_config": agents_config
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading hierarchical workflow config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading hierarchical workflow config: {str(e)}")


@workflow_router.post("/workflow/hierarchical-config/{workflow_id}")
async def save_hierarchical_workflow_config(
    workflow_id: str,
    request: Request,
    db: Session = Depends(get_db)):
    """
    Save hierarchical workflow configuration to database.
    Updates the workflow agent's config with the provided nodes, edges, and other settings.
    """
    try:
        from db.crud.agent import update_agent, AgentUpdate
        
        data = await request.json()
        
        # Validate required fields
        if "workflow_name" not in data:
            raise HTTPException(status_code=400, detail="Missing required field: workflow_name")
        if "nodes" not in data or not isinstance(data["nodes"], list):
            raise HTTPException(status_code=400, detail="Missing or invalid field: nodes (must be an array)")
        if "edges" not in data or not isinstance(data["edges"], list):
            raise HTTPException(status_code=400, detail="Missing or invalid field: edges (must be an array)")
        
        # Get existing workflow agent
        workflow_agent = get_agent_by_id(db, workflow_id)
        if not workflow_agent:
            raise HTTPException(status_code=404, detail=f"Workflow with ID {workflow_id} not found")
        
        if getattr(workflow_agent, 'type', None) != 'hierarchical_workflow':
            raise HTTPException(status_code=400, detail="Agent is not a hierarchical workflow agent")
        
        # Build the updated config
        existing_config = workflow_agent.config or {}
        
        # Update the workflow section with the new hierarchical config
        updated_config = existing_config.copy()
        updated_config['workflow'] = {
            'workflow_name': data.get('workflow_name'),
            'state_schema': data.get('state_schema', existing_config.get('workflow', {}).get('state_schema', {})),
            'nodes': data.get('nodes', []),
            'edges': data.get('edges', []),
            'entry_point': data.get('entry_point', existing_config.get('workflow', {}).get('entry_point')),
        }
        
        # Preserve any additional fields from the original workflow config
        original_workflow = existing_config.get('workflow', {})
        for key in original_workflow:
            if key not in updated_config['workflow']:
                updated_config['workflow'][key] = original_workflow[key]
        
        # Update the agent in database
        agent_update = AgentUpdate(
            name=data.get('workflow_name', workflow_agent.name),
            config=updated_config
        )
        
        updated_agent = update_agent(db, workflow_id, agent_update)
        
        if not updated_agent:
            raise HTTPException(status_code=500, detail="Failed to update workflow in database")
        
        logger.info(f"Hierarchical workflow config saved to database: {data.get('workflow_name')} (ID: {workflow_id})")
        
        return JSONResponse({
            "message": "Workflow configuration saved successfully to database",
            "workflow_id": workflow_id,
            "workflow_name": data.get('workflow_name')
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving hierarchical workflow config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving hierarchical workflow config: {str(e)}")


@workflow_router.post("/workflow/render")
async def render_workflow_graph(
    request: Request,
    db: Session = Depends(get_db)):
    """Render workflow graph as PNG image"""
    temp_workflow_path = None
    temp_agents_path = None
    
    try:
        data = await request.json()
        workflow_config = data.get("workflow_config")
        agents_config = None

        if not workflow_config:
            config_path = str(_WORKFLOW_DEFAULTS_DIR / "workflow_config.json")
            if not os.path.exists(config_path):
                raise HTTPException(status_code=404, detail="Workflow config not found")
            with open(config_path, 'r') as f:
                workflow_config = json.load(f)

        if "nodes" not in workflow_config or "edges" not in workflow_config:
            raise HTTPException(status_code=400, detail="Invalid workflow config: missing nodes or edges")

        # Extract agent IDs from the provided workflow_config (from editor) and load from database
        nodes = workflow_config.get("nodes", [])
        agents_config = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            agent_id = node.get("agent")
            if not agent_id:
                continue
            
            # Look up agent in database
            agent_obj = get_agent_by_id(db, agent_id)
            if not agent_obj:
                raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found in database")
            
            # Build agent config similar to load_workflow_and_agents_from_db
            agent_conf = dict(agent_obj.config or {})
            
            # Determine agent type
            valid_workflow_types = ["simple", "llm", "llm_with_tools"]
            if "type" not in agent_conf:
                mcp_tools = agent_conf.get("mcp", {}).get("tools", [])
                direct_tools = agent_conf.get("tools", [])
                tools_list = mcp_tools or direct_tools
                agent_type = getattr(agent_obj, "type", None)
                if agent_type and agent_type in valid_workflow_types:
                    agent_conf["type"] = agent_type
                else:
                    agent_conf["type"] = "llm_with_tools" if tools_list else "llm"
            
            # Add prompt if missing
            if agent_conf.get("type") in ["llm", "llm_with_tools"]:
                if "prompt" not in agent_conf:
                    if getattr(agent_obj, "prompt_template", None):
                        agent_conf["prompt"] = agent_obj.prompt_template
                    else:
                        agent_conf["prompt"] = ""
            
            # Add output_fields if missing
            if "output_fields" not in agent_conf:
                agent_conf["output_fields"] = ["messages"]
            
            # Add tools for llm_with_tools
            if agent_conf.get("type") == "llm_with_tools":
                mcp_tools = agent_conf.get("mcp", {}).get("tools", [])
                if mcp_tools:
                    agent_conf["tools"] = mcp_tools
            
            agents_config[str(agent_obj.id)] = agent_conf
        
        logger.info(f"Loaded {len(agents_config)} agents from database based on editor workflow config")

        # Use database agents if available, otherwise fall back to file
        if agents_config:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_agents:
                json.dump(agents_config, tmp_agents, indent=2)
                temp_agents_path = tmp_agents.name
            agents_config_path = temp_agents_path
        else:
            agents_config_path = str(_WORKFLOW_DEFAULTS_DIR / "agents_config.json")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(workflow_config, tmp_file, indent=2)
            temp_workflow_path = tmp_file.name

        try:
            builder = ConfigDrivenWorkflowBuilder(
                agents_config_path=agents_config_path,
                workflow_config_path=temp_workflow_path
            )

            builder.load_configs()

            try:
                builder.initialize_llm()
            except Exception as e:
                logger.warning(f"LLM initialization failed (may not be needed for graph): {str(e)}")

            try:
                await builder.load_tools()
            except Exception as e:
                logger.warning(f"Tool loading failed (continuing anyway): {str(e)}")

            app = await builder.build_workflow()

            try:
                graph_image = app.get_graph().draw_mermaid_png()
                png_base64 = base64.b64encode(graph_image).decode('utf-8')
                return JSONResponse({"success": True, "image": f"data:image/png;base64,{png_base64}", "message": "Workflow graph rendered successfully"})
            except Exception as e:
                logger.error(f"Error generating graph PNG: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate graph image: {str(e)}")
        finally:
            try:
                if temp_workflow_path:
                    os.unlink(temp_workflow_path)
            except:
                pass
            try:
                if temp_agents_path:
                    os.unlink(temp_agents_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering workflow graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error rendering workflow graph: {str(e)}")


@workflow_router.post("/workflow/execute-stream")
async def execute_workflow_stream(
    request: Request):
    """Execute workflow with user query and stream logs in real-time"""
    import tempfile

    data = await request.json()
    query = data.get("query", "")
    user_name = data.get("user_name", "User")
    context = data.get("context", "")
    workflow_config = data.get("workflow_config")
    agents_config = None
    chat_history = data.get("chat_history", [])

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    workflow_id = data.get("workflow_id")
    use_db_first = os.getenv("WORKFLOW_DB_FIRST", "true").lower() == "true"
    fallback_to_files = os.getenv("WORKFLOW_FILE_FALLBACK", "false").lower() == "true"

    if use_db_first and workflow_id:
        try:
            db = next(get_db())
            workflow_config, agents_config = load_workflow_and_agents_from_db(db, workflow_id)
            logger.info(f"📥 Received workflow config (DB): {workflow_config.get('workflow_name', 'unknown')} with {len(workflow_config.get('nodes', []))} nodes and {len(workflow_config.get('edges', []))} edges")
        except HTTPException:
            if not fallback_to_files and not workflow_config:
                raise
        except Exception as e:
            logger.error(f"DB workflow load failed: {e}")
            if not fallback_to_files and not workflow_config:
                raise HTTPException(status_code=500, detail="Failed to load workflow from database")

    if not workflow_config:
        if not fallback_to_files:
            raise HTTPException(status_code=400, detail="Workflow config not provided and DB lookup disabled or failed")
        config_path = str(_WORKFLOW_DEFAULTS_DIR / "workflow_config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Workflow config not found")
        with open(config_path, 'r') as f:
            workflow_config = json.load(f)

    logger.info(f"📥 Using workflow config: {workflow_config.get('workflow_name', 'unknown')} with {len(workflow_config.get('nodes', []))} nodes and {len(workflow_config.get('edges', []))} edges")

    temp_agents_path = None
    if agents_config is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_agents:
            json.dump(agents_config, tmp_agents, indent=2)
            temp_agents_path = tmp_agents.name
    agents_config_path = temp_agents_path or str(_WORKFLOW_DEFAULTS_DIR / "agents_config.json")

    if not isinstance(workflow_config, dict):
        raise HTTPException(status_code=400, detail="Invalid workflow_config: must be a dictionary")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(workflow_config, tmp_file, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_workflow_path = tmp_file.name

    try:
        with open(temp_workflow_path, 'r') as verify_file:
            verify_data = json.load(verify_file)
            logger.info(f"✓ Verified temp workflow file: {verify_data.get('workflow_name', 'unknown')} with {len(verify_data.get('nodes', []))} nodes")
    except Exception as e:
        logger.error(f"❌ Failed to verify temp workflow file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create temporary workflow file: {e}")

    async def stream_logs():
        try:
            yield f"data: {json.dumps({'type': 'log', 'message': '=' * 80})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': 'Workflow Execution Request'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': '=' * 80})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'User: {user_name}'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'Query: {query}'})}\n\n"
            context_preview = context[:100] if context else "None"
            yield f"data: {json.dumps({'type': 'log', 'message': f'Context: {context_preview}...'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': '-' * 80})}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 1: Creating workflow builder...'})}\n\n"
            builder = ConfigDrivenWorkflowBuilder(
                agents_config_path=agents_config_path,
                workflow_config_path=temp_workflow_path
            )

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 2: Loading configurations...'})}\n\n"
            builder.load_configs()
            agents_count = len(builder.agents_config)
            workflow_name = builder.workflow_config['workflow_name']
            yield f"data: {json.dumps({'type': 'log', 'message': f'✓ Loaded agent configurations: {agents_count} agents'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'✓ Loaded workflow configuration: {workflow_name}'})}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 3: Initializing LLM...'})}\n\n"
            builder.initialize_llm()
            yield f"data: {json.dumps({'type': 'log', 'message': '✓ Initialized LLM'})}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 4: Loading tools from MCP server...'})}\n\n"
            await builder.load_tools()
            yield f"data: {json.dumps({'type': 'log', 'message': f'✓ Loaded {len(builder.tools_registry)} tools from MCP server'})}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 5: Building workflow graph...'})}\n\n"
            app = await builder.build_workflow()
            yield f"data: {json.dumps({'type': 'log', 'message': '✓ Workflow graph compiled successfully'})}\n\n"

            initial_messages = []
            for msg in chat_history:
                if msg.get("role") == "user":
                    initial_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    initial_messages.append(AIMessage(content=msg.get("content", "")))
            initial_messages.append(HumanMessage(content=query))

            if chat_history:
                yield f"data: {json.dumps({'type': 'log', 'message': f'📝 Chat history: {len(chat_history)} previous messages included'})}\n\n"

            input_data = {
                "user_name": user_name,
                "query": query,
                "context": context,
                "messages": initial_messages
            }

            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 6: Executing workflow...'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': '-' * 80})}\n\n"

            result = None
            async for event in app.astream(input_data):
                for node_name, node_output in event.items():
                    yield f"data: {json.dumps({'type': 'log', 'message': f'  → Executing node: {node_name}'})}\n\n"
                    if isinstance(node_output, dict):
                        if 'messages' in node_output and node_output['messages']:
                            last_msg = node_output['messages'][-1]
                            if hasattr(last_msg, 'content'):
                                content_preview = last_msg.content[:100] if last_msg.content else ""
                                yield f"data: {json.dumps({'type': 'log', 'message': f'    Output preview: {content_preview}...'})}\n\n"
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                yield f"data: {json.dumps({'type': 'log', 'message': f'    Tool calls: {len(last_msg.tool_calls)} call(s)'})}\n\n"
                    if node_output:
                        result = node_output

            if result is None:
                yield f"data: {json.dumps({'type': 'log', 'message': '  → Getting final workflow result (invoke)...'})}\n\n"
                result = await app.ainvoke(input_data)

            yield f"data: {json.dumps({'type': 'log', 'message': '-' * 80})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': 'Step 7: Extracting final response...'})}\n\n"

            final_response = ""
            if 'messages' in result and result['messages']:
                last_message = result['messages'][-1]
                final_response = getattr(last_message, 'content', str(last_message))
                yield f"data: {json.dumps({'type': 'log', 'message': f'Final response length: {len(final_response)} characters'})}\n\n"
            else:
                final_response = "No response generated from workflow"
                yield f"data: {json.dumps({'type': 'log', 'message': '⚠ No messages in workflow result'})}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'message': '=' * 80})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': 'Workflow execution completed successfully'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': '=' * 80})}\n\n"

            yield f"data: {json.dumps({'type': 'result', 'response': final_response, 'success': True})}\n\n"

        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'log', 'message': f'❌ Error: {error_msg}'})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
        finally:
            try:
                os.unlink(temp_workflow_path)
            except:
                pass
            if temp_agents_path:
                try:
                    os.unlink(temp_agents_path)
                except:
                    pass

    return StreamingResponse(
        stream_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@workflow_router.post("/workflow/execute")
async def execute_workflow(
    request: Request):
    """Execute workflow with user query"""
    import tempfile

    try:
        data = await request.json()
        query = data.get("query", "")
        user_name = data.get("user_name", "User")
        context = data.get("context", "")
        workflow_config = data.get("workflow_config")
        agents_config = None

        logger.info("=" * 80)
        logger.info("Workflow Execution Request")
        logger.info("=" * 80)
        logger.info(f"User: {user_name}")
        logger.info(f"Query: {query}")
        logger.info(f"Context: {context[:100] if context else 'None'}...")
        logger.info("-" * 80)

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        workflow_id = data.get("workflow_id")
        use_db_first = os.getenv("WORKFLOW_DB_FIRST", "true").lower() == "true"
        fallback_to_files = os.getenv("WORKFLOW_FILE_FALLBACK", "false").lower() == "true"

        temp_workflow_path = None
        temp_agents_path = None

        if use_db_first and workflow_id:
            try:
                db = next(get_db())
                workflow_config, agents_config = load_workflow_and_agents_from_db(db, workflow_id)
                logger.info(f"Workflow (DB): {workflow_config.get('workflow_name', 'unknown')}")
            except HTTPException:
                if not fallback_to_files and not workflow_config:
                    raise
            except Exception as e:
                logger.error(f"DB workflow load failed: {e}")
                if not fallback_to_files and not workflow_config:
                    raise HTTPException(status_code=500, detail="Failed to load workflow from database")

        if not workflow_config:
            if not fallback_to_files:
                raise HTTPException(status_code=400, detail="Workflow config not provided and DB lookup disabled or failed")
            config_path = str(_WORKFLOW_DEFAULTS_DIR / "workflow_config.json")
            if not os.path.exists(config_path):
                raise HTTPException(status_code=404, detail="Workflow config not found")
            with open(config_path, 'r') as f:
                workflow_config = json.load(f)

        logger.info(f"Workflow: {workflow_config.get('workflow_name', 'unknown')}")

        if agents_config is not None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_agents:
                json.dump(agents_config, tmp_agents, indent=2)
                temp_agents_path = tmp_agents.name
        agents_config_path = temp_agents_path or str(_WORKFLOW_DEFAULTS_DIR / "agents_config.json")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(workflow_config, tmp_file, indent=2)
            temp_workflow_path = tmp_file.name

        try:
            logger.info("Step 1: Creating workflow builder...")
            builder = ConfigDrivenWorkflowBuilder(
                agents_config_path=agents_config_path,
                workflow_config_path=temp_workflow_path
            )

            logger.info("Step 2: Loading configurations...")
            builder.load_configs()

            logger.info("Step 3: Initializing LLM...")
            builder.initialize_llm()

            logger.info("Step 4: Loading tools from MCP server...")
            await builder.load_tools()

            logger.info("Step 5: Building workflow graph...")
            app = await builder.build_workflow()

            input_data = {
                "user_name": user_name,
                "query": query,
                "context": context,
                "messages": []
            }

            logger.info("Step 6: Executing workflow...")
            logger.info("-" * 80)

            result = None
            try:
                async for event in app.astream(input_data):
                    for node_name, node_output in event.items():
                        logger.info(f"  → Executing node: {node_name}")
                        if isinstance(node_output, dict):
                            if 'messages' in node_output and node_output['messages']:
                                last_msg = node_output['messages'][-1]
                                if hasattr(last_msg, 'content'):
                                    content_preview = last_msg.content[:100] if last_msg.content else ""
                                    logger.info(f"    Output preview: {content_preview}...")
                                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    logger.info(f"    Tool calls: {len(last_msg.tool_calls)} call(s)")
                            if node_output:
                                result = node_output
            except Exception as stream_error:
                logger.warning(f"Streaming failed, using invoke: {stream_error}")
                logger.info("  → Executing workflow (invoke)...")
                result = await app.ainvoke(input_data)

            logger.info("-" * 80)
            logger.info("Step 7: Extracting final response...")

            final_response = ""
            if 'messages' in result and result['messages']:
                last_message = result['messages'][-1]
                final_response = getattr(last_message, 'content', str(last_message))
                logger.info(f"Final response length: {len(final_response)} characters")
            else:
                final_response = "No response generated from workflow"
                logger.warning("No messages in workflow result")

            logger.info("=" * 80)
            logger.info("Workflow execution completed successfully")
            logger.info("=" * 80)

            return JSONResponse({
                "success": True,
                "response": final_response,
                "result": {
                    "user_name": result.get("user_name", user_name),
                    "query": result.get("query", query),
                    "context": result.get("context", context)
                }
            })

        finally:
            try:
                os.unlink(temp_workflow_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")


@workflow_router.post("/workflow/execute-hierarchical-workflow")
async def execute_hierarchical_workflow(
    request: Request,
    db: Session = Depends(get_db)):
    """
    Execute hierarchical workflow with user query - Database-only (no file fallback).
    Requires workflow_id in request body.
    Supports session_id for conversation continuity and automatic chat_history management.
    """
    transaction = None
    transaction_id = None
    session_id = None

    try:
        data = await request.json()
        query = data.get("query", "")
        user_name = data.get("user_name", "User")
        context = data.get("context", "")
        workflow_id = data.get("workflow_id")
        session_id = data.get("session_id")

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id is required")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        session_id = AgentDatabaseManager.ensure_session(db, session_id)
        transaction = AgentDatabaseManager.create_transaction(db, session_id, user_name, query, workflow_id)
        transaction_id = transaction.id

        workflow_config, agents_config = load_workflow_and_agents_from_db(db, workflow_id)

        if "state_schema" not in workflow_config:
            raise HTTPException(status_code=400, detail="Workflow config missing required field: state_schema")
        if "nodes" not in workflow_config or "edges" not in workflow_config:
            raise HTTPException(status_code=400, detail="Workflow config missing nodes or edges")

        db_chat_history = AgentDatabaseManager.load_workflow_history(db, workflow_id, session_id, limit=10)

        logger.info("=" * 80)
        logger.info(f"[Hierarchical Workflow] {workflow_config.get('workflow_name', 'unknown')}")
        logger.info(f"Query: {query}")
        logger.info(f"Session: {session_id} | Transaction: {transaction_id}")
        logger.info(f"Agents: {len(agents_config)} | Nodes: {len(workflow_config.get('nodes', []))} | Edges: {len(workflow_config.get('edges', []))}")
        if db_chat_history:
            logger.info(f"Chat History: {len(db_chat_history)} previous interactions loaded from database")
        logger.info("-" * 80)

        builder = HierarchicalWorkflowBuilder(
            agents_config=agents_config,
            workflow_config=workflow_config
        )

        builder.initialize_llm()
        await builder.load_tools()
        app_graph = await builder.build_workflow()

        node_agent_map = build_node_agent_map(workflow_config)
        # Set of node ids whose type is tool_node (for tool execution logging).
        tool_node_ids = {
            n["id"] for n in workflow_config.get("nodes", [])
            if isinstance(n, dict) and n.get("type") == "tool_node"
        }
        from utils.media_storage import get_media_storage
        storage = get_media_storage()
        initial_messages = build_input_messages(db_chat_history, query, storage=storage)

        input_data = {
            "user_name": user_name,
            "query": query,
            "context": context,
            "messages": initial_messages
        }

        logger.info("Executing workflow...")
        logger.info("-" * 80)

        result = None
        executed_nodes = []
        tool_call_inputs = {}
        last_agent_id = None
        try:
            async for event in app_graph.astream(input_data):
                for node_name, node_output in event.items():
                    executed_nodes.append(node_name)
                    logger.info(f"  → Executing node: {node_name}")
                    agent_id_for_node = node_agent_map.get(node_name) or last_agent_id
                    if node_agent_map.get(node_name):
                        last_agent_id = node_agent_map.get(node_name)
                    node_start = datetime.now()
                    content_preview = None
                    tool_calls_count = None
                    last_msg_content = None
                    if isinstance(node_output, dict):
                        if 'messages' in node_output and node_output['messages']:
                            last_msg = node_output['messages'][-1]
                            if hasattr(last_msg, 'content'):
                                last_msg_content = last_msg.content or ""
                                content_preview = last_msg_content[:100]
                                logger.info(f"    Output preview: {content_preview}...")
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                tool_calls_count = len(last_msg.tool_calls)
                                logger.info(f"    Tool calls: {tool_calls_count} call(s)")
                                for tc in last_msg.tool_calls:
                                    tool_name = tc.get("name") or tc.get("function", {}).get("name")
                                    tool_input = tc.get("args") or tc.get("arguments") or tc
                                    tool_call_id = tc.get("id") or tc.get("tool_call_id")
                                    logger.info(f"    ↳ Tool call detected: name={tool_name}, id={tool_call_id}, input_preview={(str(tool_input)[:120] if tool_input else '')}")
                                    if tool_call_id:
                                        tool_call_inputs[tool_call_id] = tool_input
                        if node_name in tool_node_ids and 'messages' in node_output and node_output['messages']:
                            for tool_msg in node_output['messages']:
                                try:
                                    tool_name = getattr(tool_msg, "name", None)
                                    tool_output = getattr(tool_msg, "content", "") or ""
                                    tool_call_id = getattr(tool_msg, "tool_call_id", None)
                                    tool_input = tool_call_inputs.get(tool_call_id) if tool_call_id else None
                                    if not tool_input:
                                        tool_input = getattr(tool_msg, "additional_kwargs", {}) or {}
                                    logger.info(f"    ↳ Tool node message: tool={tool_name}, id={tool_call_id}, input_preview={(str(tool_input)[:120] if tool_input else '')}, output_preview={(tool_output[:120] if tool_output else '')}")
                                    if tool_name and agent_id_for_node:
                                        now_ts = datetime.now()
                                        log_agent_step_langgraph(
                                            tool_name=tool_name,
                                            tool_input=str(tool_input),
                                            tool_output=str(tool_output),
                                            start_time=now_ts,
                                            end_time=now_ts,
                                            db=db,
                                            transaction_id=transaction_id,
                                            agent_id=uuid.UUID(agent_id_for_node),
                                            tool_usage=[],
                                            session_id=session_id
                                        )
                                except Exception as e:
                                    logger.warning(f"Failed to log tool output for tool node: {e}")
                            if agent_id_for_node is None:
                                logger.warning("Tool node logging skipped: no agent_id available for tool node (last_agent_id missing)")
                            if not any(getattr(msg, 'name', None) for msg in node_output.get('messages', [])):
                                logger.warning("Tool node messages missing tool name; skipping log")
                        if node_output:
                            result = node_output
                    # Log agent execution only for agent nodes, not for tool nodes
                    if agent_id_for_node and node_name not in tool_node_ids:
                        try:
                            end_time_node = datetime.now()
                            duration_ms = int((end_time_node - node_start).total_seconds() * 1000)
                            create_agent_execution_log(db, {
                                "transaction_id": transaction_id,
                                "agent_id": uuid.UUID(agent_id_for_node),
                                "start_time": node_start,
                                "end_time": end_time_node,
                                "duration_ms": duration_ms,
                                "input_data": {
                                    "query": query,
                                    "context": context,
                                    "messages_count": len(input_data.get("messages", [])),
                                    "node": node_name
                                },
                                "output_data": {
                                    "preview": content_preview,
                                    "tool_calls": tool_calls_count,
                                    "last_message": last_msg_content
                                },
                                "error_message": None
                            })
                            if last_msg_content is not None:
                                log_agent_interactions(
                                    db=db,
                                    session_id=session_id,
                                    agent_id=uuid.UUID(agent_id_for_node),
                                    query=query,
                                    result=last_msg_content
                                )
                            # LLM usage: when node_output has AIMessage with response_metadata (e.g. token_usage, model_name)
                            if isinstance(node_output, dict) and node_output.get("messages"):
                                _last = node_output["messages"][-1]
                                if isinstance(_last, AIMessage):
                                    rm = getattr(_last, "response_metadata", None) or {}
                                    tu = rm.get("token_usage") or rm.get("usage") or {}
                                    tot = tu.get("total_tokens")
                                    if tot is None and (tu.get("prompt_tokens") is not None or tu.get("completion_tokens") is not None):
                                        tot = (tu.get("prompt_tokens") or 0) + (tu.get("completion_tokens") or 0)
                                    llm_meta = {
                                        "model_name": rm.get("model_name") or rm.get("model") or "unknown",
                                        "model_provider": rm.get("model_provider") or "openai",
                                        "temperature": None,
                                        "max_tokens": tu.get("max_tokens"),
                                        "total_tokens": tot or 0,
                                        "latency_ms": duration_ms,
                                    }
                                    try:
                                        log_llm_usage(
                                            db=db,
                                            transaction_id=transaction_id,
                                            agent_id=uuid.UUID(agent_id_for_node),
                                            llm_metadata=llm_meta,
                                            query=query,
                                            result=last_msg_content or "",
                                            start_time=node_start,
                                            end_time=end_time_node,
                                        )
                                    except Exception as llm_log_err:
                                        logger.warning(f"Failed to log LLM usage for node {node_name}: {llm_log_err}")
                        except Exception as e:
                            logger.warning(f"Failed to log agent execution for node {node_name}: {e}")

            if result is None:
                logger.info("  → Getting final workflow result (invoke)...")
                result = await app_graph.ainvoke(input_data)
        except Exception as stream_error:
            logger.warning(f"Streaming failed, using invoke: {stream_error}")
            logger.info("  → Executing workflow (invoke)...")
            result = await app_graph.ainvoke(input_data)

        logger.info("-" * 80)

        final_response = ""
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            final_response = getattr(last_message, "content", str(last_message))
        else:
            final_response = "No response generated from workflow"
            logger.warning("No messages in workflow result")

        logger.info(f"Executed nodes: {' → '.join(executed_nodes)}")
        logger.info(f"Response: {len(final_response)} characters")
        logger.info("=" * 80)

        try:
            log_agent_interactions(
                db=db,
                session_id=session_id,
                agent_id=uuid.UUID(workflow_id),
                query=query,
                result=final_response
            )
        except Exception as e:
            logger.warning(f"Failed to save chat_history to database: {e}")

        try:
            transaction.end_time = datetime.now()
            transaction.status = "completed"
            transaction.final_output = {"response": final_response}
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to update transaction status: {e}")

        return JSONResponse({
            "success": True,
            "response": final_response,
            "session_id": session_id,
            "transaction_id": str(transaction_id),
            "result": {
                "user_name": result.get("user_name", user_name),
                "query": result.get("query", query),
                "context": result.get("context", context)
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing hierarchical workflow: {str(e)}")
        try:
            if transaction and db:
                transaction.status = "failed"
                transaction.end_time = datetime.now()
                transaction.final_output = {"error": str(e)}
                db.commit()
        except Exception as update_error:
            logger.warning(f"Failed to update transaction status: {update_error}")
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")
    finally:
        try:
            db.close()
        except:
            pass

async def get_and_run_workflow_agent(
    db: Session,
    workflow_id: str,
    query: str,
    transaction_id: uuid.UUID,
) -> str:
    """
    Execute a workflow agent by orchestrating its constituent sub-agents (LangGraph).
    """
    try:
        logger.info(f"Executing workflow {workflow_id} (langgraph)")
        
        # 1. Get workflow agent data
        workflow_data = AgentDatabaseManager.get_agent_data(db, workflow_id)
        session_id = AgentDatabaseManager.get_session_from_transaction(db, transaction_id)
        
        # 2. Extract workflow configuration
        workflow_config = workflow_data['config'].get('workflow', {})
        sub_agents_config = workflow_config.get('agents', [])
        
        if not sub_agents_config:
            raise HTTPException(status_code=400, detail="No sub-agents configured in workflow")
        
        logger.info(f"Found {len(sub_agents_config)} sub-agents in workflow")
        
        return await execute_langgraph_workflow(
            db, workflow_data, sub_agents_config, query, transaction_id, session_id
        )
    
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        raise


@workflow_router.post("/ask-workflow")
async def ask_workflow(
    request_data: WorkflowQueryRequest,
    db: Session = Depends(get_db)):
    """
    Execute a workflow agent with the given query.
    The workflow will coordinate its component agents using the specified framework.
    """
    try:
        logger.info(f"Workflow query request: {request_data}")
        
        # Handle session_id the same way as /ask API
        if request_data.session_id is None or request_data.session_id == "None":
            from db.crud.session import create_session
            
            session_id = uuid.uuid4()
            session = create_session(db, {
                "id": session_id,
                "start_time": datetime.now(),
                "status": "active"
            })
            request_data.session_id = str(session_id)
            logger.info(f"Created new session: {session_id}")
        else:
            session_id = request_data.session_id
        
        # Validate workflow agent exists and is actually a workflow
        workflow_agent = get_agent_by_id(db, request_data.workflow_id)
        if not workflow_agent:
            raise HTTPException(status_code=404, detail="Workflow agent not found")
        
        # Check if it's actually a workflow agent (hierarchical or supervisor)
        valid_workflow_types = ["hierarchical_workflow", "supervisor_workflow", "workflow"]
        if getattr(workflow_agent, 'type', None) not in valid_workflow_types:
            raise HTTPException(status_code=400, detail="Agent is not a workflow agent")
        
        fw = (request_data.framework or "langgraph").lower()
        if fw != "langgraph":
            raise HTTPException(status_code=400, detail="Only 'langgraph' is supported")

        # Create a pseudo QueryRequest object for compatibility with existing transaction logging
        from api.schemas.requests import QueryRequest
        query_obj = QueryRequest(
            query=request_data.query,
            agent=request_data.workflow_id,
            framework="langgraph",
            session_id=str(session_id) if session_id else None
        )
        
        # Create transaction for tracking
        transaction_id = await log_transaction(db, query_obj, None, request_data.workflow_id)
        
        # Execute the workflow agent using LangGraph
        result = await get_and_run_workflow_agent(
            db,
            request_data.workflow_id,
            request_data.query,
            uuid.UUID(transaction_id),
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Workflow agent returned no result")
        
        # Update transaction with result
        transaction = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
        if transaction:
            transaction.final_output = result
            transaction.status = "completed"
            transaction.end_time = datetime.now()
            db.commit()
        
        logger.info(f"Workflow {request_data.workflow_id} executed successfully")
        
        return {
            "result": result,
            "workflow_id": request_data.workflow_id,
            "transaction_id": str(transaction_id),
            "framework": "langgraph",
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        # Update transaction to failed status if it exists
        if 'transaction_id' in locals():
            try:
                transaction = db.query(Transaction).filter(Transaction.id == uuid.UUID(transaction_id)).first()
                if transaction:
                    transaction.status = "failed"
                    transaction.end_time = datetime.now()
                    transaction.final_output = f"Error: {str(e)}"
                    db.commit()
            except Exception as update_error:
                logger.error(f"Error updating failed transaction: {update_error}")
        
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
