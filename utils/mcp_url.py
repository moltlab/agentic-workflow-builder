"""Normalize MCP server base URLs for HTTP metadata vs SSE transport."""

from __future__ import annotations


def strip_trailing_slash(url: str) -> str:
    return (url or "").rstrip("/")


def mcp_base_for_tools_http(base_url: str) -> str:
    """Base host for GET /v1/tools (no trailing /mcp)."""
    u = strip_trailing_slash(base_url)
    if u.endswith("/mcp"):
        return u[: -len("/mcp")]
    return u


def sse_url(base_url: str) -> str:
    """SSE endpoint for CrewAI / LangChain adapters."""
    return f"{strip_trailing_slash(base_url)}/sse"
