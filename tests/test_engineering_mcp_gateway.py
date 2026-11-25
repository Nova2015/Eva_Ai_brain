import sqlite3

import pytest

from src.data_governance import register_import, set_permissions
from src.engineering import mcp_gateway, tool_registry


def _enable_blender_mcp(tmp_path, monkeypatch):
    db_path = tmp_path / "engineering.db"
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(db_path))
    tool_registry.list_tools()
    server_script = tmp_path / "blender_mcp_server.py"
    server_script.write_text("# reviewed local MCP adapter placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        "blender_mcp", executable_path=str(server_script), actor_id="owner-1"
    )
    tool_registry.set_tool_enabled(
        "blender_mcp", enabled=True, actor_id="owner-1", approved=True
    )
    return db_path


def test_mcp_tools_are_default_deny_and_require_source_review_and_approval(tmp_path, monkeypatch):
    _enable_blender_mcp(tmp_path, monkeypatch)
    assert mcp_gateway.list_mcp_tools("blender_mcp") == []

    with pytest.raises(PermissionError, match="Source review"):
        mcp_gateway.register_mcp_tool(
            "blender_mcp", "scene_create", actor_id="owner-1",
            source_reviewed=False, approved=True,
        )

    registered = mcp_gateway.register_mcp_tool(
        "blender_mcp", "scene_create", actor_id="owner-1",
        description="Create a Blender scene through a local adapter.",
        source_reviewed=True, approved=True,
    )
    assert registered["enabled"] is False
    with pytest.raises(PermissionError, match="not allowlisted"):
        mcp_gateway.authorize_mcp_call(
            "blender_mcp", "scene_create", actor_id="owner-1",
            workspace_id="cad", data_id="EVA-DATA-1", action="execute",
        )


def test_mcp_call_requires_allowlist_and_data_governance_permission(tmp_path, monkeypatch):
    db_path = _enable_blender_mcp(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "part.txt"
    source.write_text("local engineering part", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1",
    )
    set_permissions(
        record["data_id"],
        {"read": ["owner-1", "reviewer-1"], "write": ["owner-1"],
         "execute": [], "delete": []},
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
    )
    mcp_gateway.register_mcp_tool(
        "blender_mcp", "scene_create", actor_id="owner-1",
        source_reviewed=True, approved=True,
    )
    mcp_gateway.set_mcp_tool_enabled(
        "blender_mcp", "scene_create", enabled=True,
        actor_id="owner-1", approved=True,
    )

    with pytest.raises(PermissionError, match="Data Governance denied"):
        mcp_gateway.authorize_mcp_call(
            "blender_mcp", "scene_create", actor_id="reviewer-1",
            workspace_id="cad", data_id=record["data_id"], action="execute",
        )

    set_permissions(
        record["data_id"],
        {"read": ["owner-1", "reviewer-1"], "write": ["owner-1"],
         "execute": ["reviewer-1"], "delete": []},
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
    )
    allowed = mcp_gateway.authorize_mcp_call(
        "blender_mcp", "scene_create", actor_id="reviewer-1",
        workspace_id="cad", data_id=record["data_id"], action="execute",
    )
    assert allowed["authorized"] is True
    assert allowed["execution_performed"] is False

    with pytest.raises(PermissionError):
        mcp_gateway.authorize_mcp_call(
            "blender_mcp", "scene_create", actor_id="reviewer-1",
            workspace_id="other-workspace", data_id=record["data_id"], action="execute",
        )

    audit = mcp_gateway.list_mcp_audit(actor_id="reviewer-1", workspace_id="cad")
    assert [item["outcome"] for item in audit] == ["denied", "authorized"]
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE mcp_call_audit SET outcome='tampered' WHERE actor_id='reviewer-1'"
            )
    finally:
        conn.close()


def test_mcp_gateway_rejects_non_local_server_types(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    # A normal CAD application is not an MCP server and cannot be used as one.
    with pytest.raises(PermissionError, match="not enabled"):
        mcp_gateway.register_mcp_tool(
            "freecad", "arbitrary_tool", actor_id="owner-1",
            source_reviewed=True, approved=True,
        )


def test_authenticated_mcp_api_only_authorizes_and_never_invokes_server(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    _enable_blender_mcp(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "api-part.txt"
    source.write_text("MCP API governed part", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1",
    )
    mcp_gateway.register_mcp_tool(
        "blender_mcp", "scene_read", actor_id="owner-1",
        source_reviewed=True, approved=True,
    )
    mcp_gateway.set_mcp_tool_enabled(
        "blender_mcp", "scene_read", enabled=True,
        actor_id="owner-1", approved=True,
    )

    with pytest.raises(HTTPException) as error:
        routes.engineering_mcp_tools(user_id="anonymous")
    assert error.value.status_code == 401

    result = routes.engineering_authorize_mcp_call(
        routes.MCPAuthorizationRequest(
            server_id="blender_mcp", tool_name="scene_read",
            workspace_id="cad", data_id=record["data_id"], action="read",
        ),
        user_id="owner-1",
    )
    assert result["authorization"]["authorized"] is True
    assert result["execution_performed"] is False



def test_only_recorded_mcp_approver_can_enable_or_disable_tool(tmp_path, monkeypatch):
    db_path = _enable_blender_mcp(tmp_path, monkeypatch)
    mcp_gateway.register_mcp_tool(
        "blender_mcp", "scene_write", actor_id="owner-1",
        source_reviewed=True, approved=True,
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE mcp_tool_allowlist SET approved_by='reviewer-1' "
            "WHERE server_id='blender_mcp' AND tool_name='scene_write'"
        )
        conn.commit()
    finally:
        conn.close()

    for actor_id in ("owner-1", "other-user"):
        with pytest.raises(PermissionError, match="recorded MCP approver"):
            mcp_gateway.set_mcp_tool_enabled(
                "blender_mcp", "scene_write", enabled=True,
                actor_id=actor_id, approved=True,
            )
    assert mcp_gateway.get_mcp_tool("blender_mcp", "scene_write")["enabled"] is False
    enabled = mcp_gateway.set_mcp_tool_enabled(
        "blender_mcp", "scene_write", enabled=True,
        actor_id="reviewer-1", approved=True,
    )
    assert enabled["enabled"] is True
