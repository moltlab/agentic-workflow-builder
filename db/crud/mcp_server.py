"""MCP server registry CRUD."""

from __future__ import annotations

import os
import uuid as uuid_mod
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import MCPServer, Tool
from utils.logging_utils import get_logger

logger = get_logger("mcp_server_crud")


def ensure_default_mcp_server_from_env(db: Session) -> Optional[MCPServer]:
    """If no servers exist, seed one named 'default' from MCP_SERVER_URL."""
    count = db.query(MCPServer).count()
    if count > 0:
        return None
    url = os.getenv("MCP_SERVER_URL", "").strip()
    if not url:
        logger.warning("No MCP_SERVER_URL set; skipping default mcp_servers seed")
        return None
    row = MCPServer(
        name="default",
        base_url=url.rstrip("/"),
        description="Seeded from MCP_SERVER_URL",
        entity_id=None,
        is_active=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("Seeded default MCP server id=%s base_url=%s", row.id, row.base_url)
    return row


def backfill_tools_mcp_server_id(db: Session, server: MCPServer) -> int:
    n = 0
    for t in db.query(Tool).filter(Tool.mcp_server_id.is_(None)).all():
        t.mcp_server_id = server.id
        db.add(t)
        n += 1
    if n:
        db.commit()
        logger.info("Backfilled mcp_server_id for %s tool row(s)", n)
    return n


def list_servers(db: Session, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
    q = db.query(MCPServer).filter(MCPServer.is_active.is_(True))
    if entity_id is not None and str(entity_id).strip() != "":
        q = q.filter((MCPServer.entity_id == entity_id) | (MCPServer.entity_id.is_(None)))
    rows = q.order_by(MCPServer.name).all()
    return [
        {
            "id": str(r.id),
            "name": r.name,
            "base_url": r.base_url,
            "description": r.description,
            "entity_id": r.entity_id,
        }
        for r in rows
    ]


def get_by_name(db: Session, name: str) -> Optional[MCPServer]:
    return db.query(MCPServer).filter(MCPServer.name == name, MCPServer.is_active.is_(True)).first()


def get_default_server(db: Session) -> Optional[MCPServer]:
    d = get_by_name(db, "default")
    if d:
        return d
    return db.query(MCPServer).filter(MCPServer.is_active.is_(True)).first()


def create_server(
    db: Session,
    name: str,
    base_url: str,
    description: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> MCPServer:
    row = MCPServer(
        name=name.strip(),
        base_url=base_url.rstrip("/"),
        description=description,
        entity_id=entity_id,
        is_active=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def resolve_base_url_for_tool(
    tool_cfg: Dict[str, Any],
    mcp_block: Dict[str, Any],
    db: Session,
    default_url: str,
) -> str:
    """Pick MCP base URL for a tool entry (multi-server)."""
    sname = (tool_cfg.get("mcp_server") or "").strip()
    if sname:
        s = get_by_name(db, sname)
        if s:
            return s.base_url
    block_url = (mcp_block.get("server_url") or "").strip()
    if block_url:
        return block_url.rstrip("/")
    d = get_default_server(db)
    if d:
        return d.base_url
    return default_url.rstrip("/")
