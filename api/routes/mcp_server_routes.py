"""MCP server registry (control plane)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.config import get_db
from db.crud import mcp_server as mcp_server_crud
from db.models import MCPServer
from utils.logging_utils import get_logger

logger = get_logger("mcp_server_routes")

mcp_server_router = APIRouter(tags=["mcp-servers"])


class MCPServerCreate(BaseModel):
    name: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    description: Optional[str] = None


@mcp_server_router.get("/mcp-servers")
async def list_mcp_servers(
    db: Session = Depends(get_db),
):
    return JSONResponse(content=mcp_server_crud.list_servers(db, entity_id="default_entity"))


@mcp_server_router.post("/mcp-servers")
async def register_mcp_server(
    body: MCPServerCreate,
    db: Session = Depends(get_db),
):
    try:
        row = mcp_server_crud.create_server(
            db,
            name=body.name,
            base_url=body.base_url,
            description=body.description,
            entity_id="default_entity",
        )
        return JSONResponse(
            content={
                "id": str(row.id),
                "name": row.name,
                "base_url": row.base_url,
                "description": row.description,
                "entity_id": row.entity_id,
            },
            status_code=201,
        )
    except Exception as e:
        logger.error("register_mcp_server: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e


@mcp_server_router.delete("/mcp-servers/{server_id}")
async def deactivate_mcp_server(
    server_id: str,
    db: Session = Depends(get_db),
):
    from uuid import UUID

    try:
        uid = UUID(server_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    row = db.query(MCPServer).filter(MCPServer.id == uid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if row.entity_id and row.entity_id != "default_entity":
        raise HTTPException(status_code=403, detail="Forbidden")
    row.is_active = False
    db.add(row)
    db.commit()
    return JSONResponse(content={"status": "ok"})
