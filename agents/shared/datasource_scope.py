"""
Datasource catalog text and allowlist validation for MCP tools that declare
accepts_datasource_types (sql_database, vector_store, document_corpus, …) or legacy SQL name heuristics for catalog attachment.

Supports multi-datasource agent configs (`datasources` array) and legacy
single `resource_uri` / `resource_name` fields.
"""

import json
from types import MethodType
from typing import Any, Dict, FrozenSet, List, Optional

from db.config import SessionLocal
try:
    from services.credential_resolver import parse_datasource_uuid_from_uri, resolve_merged_for_mcp
except ImportError:
    # Optional dependency in stripped-down/local setups.
    def parse_datasource_uuid_from_uri(resource_uri: Any) -> Optional[str]:
        _ = resource_uri
        return None

    def resolve_merged_for_mcp(db: Any, datasource_uuid: str) -> Dict[str, Any]:
        _ = (db, datasource_uuid)
        return {}
from utils.logging_utils import get_logger

logger = get_logger("datasource_scope")

try:
    from langchain_core.tools import StructuredTool as _LCStructuredTool
except ImportError:
    _LCStructuredTool = None  # type: ignore[misc, assignment]

# LangChain passes these through tool execution; do not treat as MCP tool args.
_LC_RUN_RESERVED: FrozenSet[str] = frozenset({"run_manager", "config", "callbacks"})


def _is_lc_structured_tool(tool: Any) -> bool:
    return bool(_LCStructuredTool and isinstance(tool, _LCStructuredTool))


def _assign_tool_attr(tool: Any, name: str, value: Any) -> None:
    """Set an attribute on tools; LangChain StructuredTool rejects extra fields via Pydantic."""
    try:
        setattr(tool, name, value)
    except ValueError as e:
        if "has no field" in str(e):
            object.__setattr__(tool, name, value)
        else:
            raise


def _merge_tool_payload(primary: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge JSON/dict primary input with keyword args (Crew passes kwargs-only to _run)."""
    extra = dict(extra or {})
    if primary is None:
        parsed: Dict[str, Any] = {}
    else:
        p = _parse_tool_input_to_dict(primary)
        parsed = dict(p) if isinstance(p, dict) else {}
    return {**parsed, **extra}


def _parse_tool_input_to_dict(input_data: Any) -> Any:
    """
    CrewAI (and some MCP adapters) pass tool arguments as a JSON string.
    Normalize to dict so allowlist + credential injection run.
    """
    if isinstance(input_data, dict):
        return input_data
    if isinstance(input_data, str):
        s = input_data.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.debug("Tool input is non-JSON string, skipping dict coercion")
    return input_data


def _tool_needs_datasource_catalog(meta: Dict[str, Any], tool_name: str) -> bool:
    """True if tool should receive datasource catalog + allowlist (metadata or legacy SQL name)."""
    types = meta.get("accepts_datasource_types")
    if isinstance(types, list) and len(types) > 0:
        return True
    n = (tool_name or "").lower()
    return "sql" in n or "query_sql" in n


def _merged_tool_meta(
    tool_name: str, tool_config: Dict[str, Any], metadata_cache: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(metadata_cache.get(tool_name) or {})
    tc_types = tool_config.get("accepts_datasource_types")
    if isinstance(tc_types, list) and tc_types:
        out["accepts_datasource_types"] = tc_types
    return out


def validate_agent_mcp_datasource_bindings(config: Optional[Dict[str, Any]]) -> None:
    """
    Ensure every MCP tool that requires datasource bindings has at least one
    linked datasource (datasources list or legacy resource_uri).

    Raises ValueError with a user-facing message if validation fails.
    """
    if not config or not isinstance(config, dict):
        return
    mcp = config.get("mcp")
    if not isinstance(mcp, dict):
        return
    if not mcp.get("enabled"):
        return
    tools = mcp.get("tools")
    if not isinstance(tools, list):
        return
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        if not name:
            continue
        types = t.get("accepts_datasource_types")
        if not isinstance(types, list) or len(types) == 0:
            continue
        meta_cache: Dict[str, Any] = {name: {"accepts_datasource_types": types}}
        # Legacy inline SQL config does not use datasource:// bindings
        if "database" in t and _tool_accepts_legacy_database_dict(name, meta_cache):
            continue
        if normalize_sql_tool_datasources(t):
            continue
        raise ValueError(
            f'MCP tool "{name}" requires at least one linked datasource. '
            "Select one or more compatible datasources for this agent."
        )


def _tool_accepts_legacy_database_dict(tool_name: str, metadata_cache: Dict[str, Any]) -> bool:
    """Inline database dict injection is SQL-only."""
    meta = metadata_cache.get(tool_name) or {}
    types = meta.get("accepts_datasource_types")
    if isinstance(types, list) and "sql_database" in types:
        return True
    n = (tool_name or "").lower()
    return "sql" in n or "query_sql" in n


def normalize_sql_tool_datasources(tool_config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a list of {resource_uri, resource_name} from stored tool config."""
    raw = tool_config.get("datasources")
    if isinstance(raw, list) and raw:
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            uri = (item.get("resource_uri") or "").strip()
            ds_id = (item.get("datasource_id") or "").strip()
            if not uri and ds_id:
                uri = f"datasource://{ds_id}"
            if not uri.startswith("datasource://"):
                continue
            name = (item.get("resource_name") or uri).strip()
            out.append({"resource_uri": uri, "resource_name": name})
        return out

    legacy_uri = tool_config.get("resource_uri")
    if legacy_uri:
        uri = str(legacy_uri).strip()
        if uri and not uri.startswith("datasource://"):
            uri = f"datasource://{uri.replace('datasource://', '')}"
        name = (tool_config.get("resource_name") or uri).strip()
        return [{"resource_uri": uri, "resource_name": name}]
    return []


def allowed_resource_uris_frozen(datasources: List[Dict[str, str]]) -> FrozenSet[str]:
    return frozenset(d["resource_uri"] for d in datasources if d.get("resource_uri"))


def build_datasource_catalog_suffix(datasources: List[Dict[str, str]]) -> str:
    if not datasources:
        return ""
    lines = [
        "",
        "Available datasources for this agent (use the exact resource_uri on each call):",
        "",
    ]
    for d in datasources:
        lines.append(f"  • {d['resource_name']} → resource_uri: {d['resource_uri']}")
    lines.append("")
    lines.append(
        "You MUST pass the correct resource_uri on every call so the tool uses the intended datasource."
    )
    return "\n".join(lines)


def _secret_like_key(name: str) -> bool:
    n = name.lower()
    return any(
        x in n
        for x in (
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "credential",
            "private",
            "authorization",
            "bearer",
        )
    )


def _summarize_merged_credentials_for_log(merged: Dict[str, Any]) -> str:
    """Safe one-line summary: no secret values."""
    dt = merged.get("datasource_type", "")
    prov = merged.get("provider", "")
    ru = merged.get("resource_uri", "")
    public_keys = sorted(k for k in merged if not _secret_like_key(k))
    secret_n = len(merged) - len(public_keys)
    open_preview = public_keys[:14]
    tail = f" (+{secret_n} redacted)" if secret_n else ""
    return (
        f"datasource_type={dt!r} provider={prov!r} resource_uri={ru!r} "
        f"open_keys={open_preview}{tail}"
    )


def _inject_system_credentials(
    input_data: Any,
    *,
    tool_name: str = "unknown",
    injection_path: str = "unknown",
) -> Any:
    """
    Copy dict input and attach system_credentials for stateless MCP tools (FastMCP-safe arg name).

    Logs at INFO when credentials are resolved and attached (values are never logged).
    """
    if not isinstance(input_data, dict):
        logger.debug(
            "[MCP system_credentials] skip tool=%s path=%s reason=not_a_dict",
            tool_name,
            injection_path,
        )
        return input_data
    ru = input_data.get("resource_uri")
    ds_uuid = parse_datasource_uuid_from_uri(ru)
    if ds_uuid is None:
        logger.debug(
            "[MCP system_credentials] skip tool=%s path=%s reason=no_datasource_uuid "
            "resource_uri=%r",
            tool_name,
            injection_path,
            ru,
        )
        return input_data
    db = SessionLocal()
    try:
        merged = resolve_merged_for_mcp(db, ds_uuid)
    except Exception as e:
        logger.warning("Credential resolve failed for %s: %s", ru, e)
        return {
            "error": "Could not resolve datasource credentials",
            "resource_uri": ru,
        }
    finally:
        db.close()
    out = dict(input_data)
    out["system_credentials"] = merged
    logger.info(
        "[MCP system_credentials] attached tool=%s path=%s datasource_id=%s %s — "
        "next hop is the MCP client (tool call to mcp-server will include "
        "system_credentials in the JSON-RPC/tool arguments; values not logged)",
        tool_name,
        injection_path,
        ds_uuid,
        _summarize_merged_credentials_for_log(merged),
    )
    return out


def patch_mcp_tool_datasource_allowlist(
    tool: Any,
    allowed_uris: FrozenSet[str],
    inject_credentials: bool = True,
) -> Any:
    """
    Wrap tool invocation methods so resource_uri must be in allowed_uris.

    When inject_credentials is True, resolves framework-stored credentials and
    adds system_credentials (must match MCP tool parameter name for FastMCP).

    LangChain StructuredTool: cannot assign ``invoke``/``ainvoke`` normally (Pydantic);
    we use ``object.__setattr__`` when needed. Credentials are injected in ``_run`` /
    ``_arun`` so tool args still pass Pydantic validation (system_credentials is not
    on the LLM-facing schema).

    CrewAI MCP tools call the bound ``_run`` with keyword args only; merge kwargs
    when building the payload for injection.

    Does not block calls that omit resource_uri (MCP may use env fallback).
    Legacy database dict bypass is unchanged.
    """
    if not allowed_uris:
        return tool

    _tool_name = getattr(tool, "name", None) or "unknown"

    def _reject(resource_uri: Optional[str]) -> Optional[Dict[str, Any]]:
        if resource_uri is None:
            return None
        s = str(resource_uri).strip()
        if s not in allowed_uris:
            return {
                "error": "resource_uri not permitted for this agent",
                "resource_uri": resource_uri,
                "allowed_resource_uris": sorted(allowed_uris),
            }
        return None

    lc = _is_lc_structured_tool(tool)
    if lc:
        from langchain_core.runnables import RunnableConfig as _LCRunnableConfig
    else:
        _LCRunnableConfig = None  # unused; keeps name defined for type checkers

    if hasattr(tool, "invoke"):
        _orig_inv = tool.invoke

        def _inv(input: Any, config: Any = None, **kw: Any) -> Any:
            merged = _merge_tool_payload(input, kw)
            rej = _reject(merged.get("resource_uri"))
            if rej is not None:
                return rej
            if inject_credentials:
                logger.debug(
                    "[MCP system_credentials] tool=%s stage=invoke note=no_payload_here "
                    "(injection runs in patched _run/_arun before the MCP client sends the tool call)",
                    _tool_name,
                )
            return _orig_inv(input, config=config, **kw)

        _assign_tool_attr(tool, "invoke", _inv)

    if hasattr(tool, "ainvoke"):
        _orig_ainv = tool.ainvoke

        async def _ainv(input: Any, config: Any = None, **kw: Any) -> Any:
            merged = _merge_tool_payload(input, kw)
            rej = _reject(merged.get("resource_uri"))
            if rej is not None:
                return rej
            if inject_credentials:
                logger.debug(
                    "[MCP system_credentials] tool=%s stage=ainvoke note=no_payload_here "
                    "(injection runs in patched _run/_arun before the MCP client sends the tool call)",
                    _tool_name,
                )
            return await _orig_ainv(input, config=config, **kw)

        _assign_tool_attr(tool, "ainvoke", _ainv)

    if hasattr(tool, "_run"):
        _orig_run = tool._run

        if lc:

            def _lc_run(
                self: Any,
                *args: Any,
                config: _LCRunnableConfig,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> Any:
                # Mirror StructuredTool._run signature so BaseTool.run() injects RunnableConfig via
                # _get_runnable_config_param(self._run).
                tool_only = {k: v for k, v in kwargs.items() if k not in _LC_RUN_RESERVED}
                if args:
                    p = _parse_tool_input_to_dict(args[0])
                    if isinstance(p, dict):
                        tool_only = {**p, **tool_only}
                rej = _reject(tool_only.get("resource_uri"))
                if rej is not None:
                    return rej
                if not inject_credentials:
                    return _orig_run(
                        *args, config=config, run_manager=run_manager, **kwargs
                    )
                payload = _inject_system_credentials(
                    dict(tool_only),
                    tool_name=_tool_name,
                    injection_path="langchain.StructuredTool._run",
                )
                if isinstance(payload, dict) and payload.get("error"):
                    return payload
                pass_kw = dict(kwargs)
                pass_kw.update(payload)
                logger.info(
                    "[MCP system_credentials] forwarding tool=%s path=langchain.StructuredTool._run "
                    "→ bound MCP adapter / mcp-server (args include system_credentials key)",
                    _tool_name,
                )
                return _orig_run(
                    *args, config=config, run_manager=run_manager, **pass_kw
                )

            _assign_tool_attr(tool, "_run", MethodType(_lc_run, tool))
        else:

            def _crew_run(self: Any, *args: Any, **kwargs: Any) -> Any:
                merged = _merge_tool_payload(args[0] if args else None, kwargs)
                rej = _reject(merged.get("resource_uri"))
                if rej is not None:
                    return json.dumps(rej)
                if not inject_credentials:
                    return _orig_run(*args, **kwargs)
                payload = _inject_system_credentials(
                    dict(merged),
                    tool_name=_tool_name,
                    injection_path="crewai.BaseTool._run",
                )
                if isinstance(payload, dict) and payload.get("error"):
                    return json.dumps(payload)
                logger.info(
                    "[MCP system_credentials] forwarding tool=%s path=crewai.BaseTool._run "
                    "→ mcpadapt / MCP SSE (kwargs include system_credentials key)",
                    _tool_name,
                )
                if not args:
                    return _orig_run(**payload)
                return _orig_run(*args, **payload)

            _assign_tool_attr(tool, "_run", MethodType(_crew_run, tool))

    if hasattr(tool, "_arun"):
        _orig_arun = tool._arun

        if lc:

            async def _lc_arun(
                self: Any,
                *args: Any,
                config: _LCRunnableConfig,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> Any:
                tool_only = {k: v for k, v in kwargs.items() if k not in _LC_RUN_RESERVED}
                if args:
                    p = _parse_tool_input_to_dict(args[0])
                    if isinstance(p, dict):
                        tool_only = {**p, **tool_only}
                rej = _reject(tool_only.get("resource_uri"))
                if rej is not None:
                    return rej
                if not inject_credentials:
                    return await _orig_arun(
                        *args, config=config, run_manager=run_manager, **kwargs
                    )
                payload = _inject_system_credentials(
                    dict(tool_only),
                    tool_name=_tool_name,
                    injection_path="langchain.StructuredTool._arun",
                )
                if isinstance(payload, dict) and payload.get("error"):
                    return payload
                pass_kw = dict(kwargs)
                pass_kw.update(payload)
                logger.info(
                    "[MCP system_credentials] forwarding tool=%s path=langchain.StructuredTool._arun "
                    "→ bound MCP adapter / mcp-server (args include system_credentials key)",
                    _tool_name,
                )
                return await _orig_arun(
                    *args, config=config, run_manager=run_manager, **pass_kw
                )

            _assign_tool_attr(tool, "_arun", MethodType(_lc_arun, tool))
        else:

            async def _crew_arun(self: Any, *args: Any, **kwargs: Any) -> Any:
                merged = _merge_tool_payload(args[0] if args else None, kwargs)
                rej = _reject(merged.get("resource_uri"))
                if rej is not None:
                    return rej
                if not inject_credentials:
                    return await _orig_arun(*args, **kwargs)
                payload = _inject_system_credentials(
                    dict(merged),
                    tool_name=_tool_name,
                    injection_path="crewai.BaseTool._arun",
                )
                if isinstance(payload, dict) and payload.get("error"):
                    return payload
                logger.info(
                    "[MCP system_credentials] forwarding tool=%s path=crewai.BaseTool._arun "
                    "→ mcpadapt / MCP SSE (kwargs include system_credentials key)",
                    _tool_name,
                )
                if not args:
                    return await _orig_arun(**payload)
                return await _orig_arun(*args, **payload)

            _assign_tool_attr(tool, "_arun", MethodType(_crew_arun, tool))

    return tool


def apply_datasource_config_to_tools(
    filtered_tools: List[Any],
    mcp_tools_config: List[Dict[str, Any]],
    tool_metadata_cache: Optional[Dict[str, Any]] = None,
) -> None:
    """Mutate tools in-place: catalog description + allowlist for datasource-capable tools."""
    cache: Dict[str, Any] = dict(tool_metadata_cache or {})

    for tool_config in mcp_tools_config:
        name = tool_config.get("name") or ""
        merged = _merged_tool_meta(name, tool_config, cache)
        if not _tool_needs_datasource_catalog(merged, name):
            continue

        if "database" in tool_config:
            if not _tool_accepts_legacy_database_dict(name, merged):
                continue
            for tool in filtered_tools:
                if tool.name == name:
                    database_config = tool_config["database"]
                    tool.description = (
                        f"{tool.description}\n\n"
                        f"Database Configuration(MUST USE IN ALL TOOL CALLS): {database_config}"
                    )
                    setattr(tool, "_database_config", database_config)
                    logger.info("Added legacy database configuration to tool '%s'", name)
            continue

        ds_list = normalize_sql_tool_datasources(tool_config)
        if not ds_list:
            continue

        catalog = build_datasource_catalog_suffix(ds_list)
        allowed = allowed_resource_uris_frozen(ds_list)

        for tool in filtered_tools:
            if tool.name != name:
                continue
            base = tool.description or ""
            tool.description = f"{base}{catalog}"
            patch_mcp_tool_datasource_allowlist(tool, allowed)
            setattr(tool, "_allowed_datasource_uris", allowed)
            setattr(tool, "_datasources_catalog", ds_list)
            logger.info(
                "Attached %d datasource(s) to tool '%s' with allowlist enforcement",
                len(ds_list),
                name,
            )
            break


# Backward-compatible alias
apply_datasource_config_to_sql_tools = apply_datasource_config_to_tools
