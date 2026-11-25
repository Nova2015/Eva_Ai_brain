from pathlib import Path

import pytest

from src.engineering import tool_registry


def test_registry_seeds_approved_engineering_candidates_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tools = tool_registry.list_tools()
    by_id = {item["tool_id"]: item for item in tools}

    for tool_id in ("freecad", "blender", "gmsh", "calculix", "openfoam",
                    "linuxcnc", "freecad_mcp", "blender_mcp", "autodesk_moldflow"):
        assert tool_id in by_id
        assert by_id[tool_id]["enabled"] is False
        assert by_id[tool_id]["validation_status"] == "not_checked"
        assert by_id[tool_id]["machine_start_allowed"] is False

    assert "python_automation" in by_id["blender"]["capabilities"]
    assert "5_axis_cam" in by_id["mastercam"]["capabilities"]
    assert "gcode_execution" in by_id["linuxcnc"]["capabilities"]


def test_tool_must_be_validated_and_approved_before_enablement(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / "freecad-stub.exe"
    executable.write_text("local executable placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="validated"):
        tool_registry.set_tool_enabled(
            "freecad", enabled=True, actor_id="owner-1", approved=True
        )
    with pytest.raises(PermissionError, match="Explicit approval"):
        tool_registry.set_tool_enabled(
            "freecad", enabled=True, actor_id="owner-1", approved=False
        )

    validated = tool_registry.validate_tool_installation(
        "freecad", executable_path=str(executable),
        actor_id="owner-1", version="test-version",
    )
    assert validated["validation_status"] == "path_verified"
    assert validated["enabled"] is False

    enabled = tool_registry.set_tool_enabled(
        "freecad", enabled=True, actor_id="owner-1", approved=True
    )
    assert enabled["enabled"] is True
    assert tool_registry.resolve_tool("freecad")["executable_path"] == str(executable.resolve())


def test_commercial_tool_requires_explicit_license_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / "catia-stub.exe"
    executable.write_text("placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        "catia", executable_path=str(executable), actor_id="owner-1"
    )

    with pytest.raises(ValueError, match="Commercial license confirmation"):
        tool_registry.set_tool_enabled(
            "catia", enabled=True, actor_id="owner-1", approved=True,
            license_confirmed=False,
        )
    enabled = tool_registry.set_tool_enabled(
        "catia", enabled=True, actor_id="owner-1", approved=True,
        license_confirmed=True,
    )
    assert enabled["enabled"] is True
    assert enabled["license_confirmed"] is True


def test_tool_resolution_fails_if_executable_disappears(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / "blender-stub.exe"
    executable.write_text("placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        "blender", executable_path=str(executable), actor_id="owner-1"
    )
    tool_registry.set_tool_enabled(
        "blender", enabled=True, actor_id="owner-1", approved=True
    )
    executable.unlink()

    with pytest.raises(PermissionError, match="no longer available"):
        tool_registry.resolve_tool("blender")


def test_registry_rejects_machine_start_capability(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    with pytest.raises(ValueError, match="prohibits automatic machine start"):
        tool_registry.register_tool(
            tool_id="unsafe_controller",
            name="Unsafe Controller",
            category="cnc_controller",
            integration_type="machine_controller",
            license_type="unknown",
            capabilities=["machine_start"],
            actor_id="owner-1",
            machine_start_allowed=True,
        )
