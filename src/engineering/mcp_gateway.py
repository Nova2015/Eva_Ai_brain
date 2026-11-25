"""Default-deny policy gateway for local MCP servers and engineering adapters.

This module authorizes and audits calls but deliberately does not invoke an MCP client.
No remote server URLs or cloud endpoints are accepted.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import tool_registry


_ALLOWED_INTEGRATIONS = frozenset({"mcp_server", "local_adapter", "wsl_adapter"})
_ALLOWED_ACTIONS = frozenset({"read", "write", "execute", "delete"})


def _connect() -> sqlite3.Connection:
    path = tool_registry._db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS mcp_tool_allowlist (
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            source_reviewed INTEGER NOT NULL DEFAULT 0,
            approved_by TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(server_id, tool_name)
        );

        CREATE TABLE IF NOT EXISTS mcp_call_audit (
            call_id TEXT PRIMARY KEY,
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            data_id TEXT,
            action TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mcp_audit_scope
            ON mcp_call_audit(actor_id, workspace_id, created_at);

        CREATE TRIGGER IF NOT EXISTS trg_mcp_call_audit_no_update
        BEFORE UPDATE ON mcp_call_audit
        BEGIN
            SELECT RAISE(ABORT, 'mcp_call_audit is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_mcp_call_audit_no_delete
        BEFORE DELETE ON mcp_call_audit
        BEGIN
            SELECT RAISE(ABORT, 'mcp_call_audit is append-only');
        END;
        """
    )
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(
    conn: sqlite3.Connection,
    *,
    server_id: str,
    tool_name: str,
    actor_id: str,
    workspace_id: str,
    data_id: Optional[str],
    action: str,
    outcome: str,
    reason: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    conn.execute(
        """INSERT INTO mcp_call_audit
           (call_id, server_id, tool_name, actor_id, workspace_id, data_id,
            action, outcome, reason, details_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "EVA-MCP-CALL-" + uuid.uuid4().hex.upper(),
            server_id, tool_name, actor_id, workspace_id, data_id,
            action, outcome, reason,
            json.dumps(details or {}, sort_keys=True), _now(),
        ),
    )


def register_mcp_tool(
    server_id: str,
    tool_name: str,
    *,
    actor_id: str,
    description: str = "",
    source_reviewed: bool,
    approved: bool,
) -> dict[str, Any]:
    for field, value in (("server_id", server_id), ("tool_name", tool_name),
                         ("actor_id", actor_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if source_reviewed is not True or approved is not True:
        raise PermissionError("Source review and explicit approval are required")
    server = tool_registry.resolve_tool(server_id)
    if server["integration_type"] not in _ALLOWED_INTEGRATIONS:
        raise ValueError("Selected tool is not registered as a local MCP/adapter integration")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO mcp_tool_allowlist
               (server_id, tool_name, description, source_reviewed,
                approved_by, approved_at, enabled, created_by, created_at)
               VALUES (?, ?, ?, 1, ?, ?, 0, ?, ?)""",
            (server_id, tool_name.strip(), description, actor_id, _now(), actor_id, _now()),
        )
    return get_mcp_tool(server_id, tool_name)


def get_mcp_tool(server_id: str, tool_name: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM mcp_tool_allowlist WHERE server_id=? AND tool_name=?",
            (server_id, tool_name),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["source_reviewed"] = bool(item["source_reviewed"])
    item["enabled"] = bool(item["enabled"])
    return item


def list_mcp_tools(server_id: Optional[str] = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if server_id:
            rows = conn.execute(
                "SELECT * FROM mcp_tool_allowlist WHERE server_id=? ORDER BY tool_name",
                (server_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM mcp_tool_allowlist ORDER BY server_id, tool_name"
            ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["source_reviewed"] = bool(item["source_reviewed"])
        item["enabled"] = bool(item["enabled"])
        results.append(item)
    return results


def set_mcp_tool_enabled(
    server_id: str,
    tool_name: str,
    *,
    enabled: bool,
    actor_id: str,
    approved: bool,
) -> dict[str, Any]:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")
    if approved is not True:
        raise PermissionError("Explicit approval is required to change MCP tool access")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    if enabled:
        server = tool_registry.resolve_tool(server_id)
        if server["integration_type"] not in _ALLOWED_INTEGRATIONS:
            raise PermissionError("Only approved local MCP adapters can be enabled")
    with _connect() as conn:
        cursor = conn.execute(
            """UPDATE mcp_tool_allowlist SET enabled=?
               WHERE server_id=? AND tool_name=? AND source_reviewed=1""",
            (int(enabled), server_id, tool_name),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"MCP tool is not reviewed or registered: {server_id}/{tool_name}")
    return get_mcp_tool(server_id, tool_name)  # type: ignore[return-value]


def authorize_mcp_call(
    server_id: str,
    tool_name: str,
    *,
    actor_id: str,
    workspace_id: str,
    data_id: str,
    action: str,
) -> dict[str, Any]:
    """Authorize one local MCP action; this function never performs the call."""
    for field, value in (("server_id", server_id), ("tool_name", tool_name),
                         ("actor_id", actor_id), ("workspace_id", workspace_id),
                         ("data_id", data_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if action not in _ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported MCP action: {action}")

    reason = "authorized"
    try:
        server = tool_registry.resolve_tool(server_id)
        if server["integration_type"] not in _ALLOWED_INTEGRATIONS:
            raise PermissionError("MCP calls are restricted to local adapters")
        allow = get_mcp_tool(server_id, tool_name)
        if allow is None or not allow["enabled"] or not allow["source_reviewed"]:
            raise PermissionError("MCP tool is not allowlisted and enabled")
        try:
            from ..data_governance import authorize_data_action
        except ImportError:
            from data_governance import authorize_data_action
        if not authorize_data_action(
            data_id, actor_id=actor_id, workspace_id=workspace_id, action=action
        ):
            raise PermissionError(f"Data Governance denied the {action} action")
    except Exception as exc:
        reason = str(exc)
        with _connect() as conn:
            _audit(
                conn, server_id=server_id, tool_name=tool_name, actor_id=actor_id,
                workspace_id=workspace_id, data_id=data_id, action=action,
                outcome="denied", reason=reason,
            )
        raise

    with _connect() as conn:
        _audit(
            conn, server_id=server_id, tool_name=tool_name, actor_id=actor_id,
            workspace_id=workspace_id, data_id=data_id, action=action,
            outcome="authorized", reason=reason,
            details={"server_integration": server["integration_type"]},
        )
    return {
        "authorized": True,
        "server_id": server_id,
        "tool_name": tool_name,
        "actor_id": actor_id,
        "workspace_id": workspace_id,
        "data_id": data_id,
        "action": action,
        "execution_performed": False,
    }


def list_mcp_audit(
    *, actor_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    for field, value in (("actor_id", actor_id), ("workspace_id", workspace_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM mcp_call_audit
               WHERE actor_id=? AND workspace_id=? ORDER BY created_at""",
            (actor_id, workspace_id),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json"))
        result.append(item)
    return result
