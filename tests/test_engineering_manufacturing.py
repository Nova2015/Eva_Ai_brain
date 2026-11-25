from pathlib import Path

import pytest

from src.engineering import manufacturing, tool_registry


def _enable_tool(tmp_path, monkeypatch, tool_id):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / f"{tool_id}.stub"
    executable.write_text("local placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        tool_id, executable_path=str(executable), actor_id="owner-1"
    )
    tool_registry.set_tool_enabled(
        tool_id, enabled=True, actor_id="owner-1", approved=True,
        license_confirmed=(tool_id in {"catia", "mastercam"}),
    )


def test_all_ten_approved_manufacturing_processes_are_registered():
    processes = manufacturing.list_processes()
    assert len(processes) == 10
    expected = {
        "injection_molding", "blow_molding", "thermoforming",
        "compression_molding", "rim_rtm", "cnc_machining", "edm",
        "additive_manufacturing", "sheet_metal_stamping", "plastic_part_assembly",
    }
    assert {item["process_id"] for item in processes} == expected
    assert all(item["machine_start_allowed"] is False for item in processes)


def test_injection_molding_validation_returns_domain_checks_and_tool_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    result = manufacturing.validate_process_request(
        "injection_molding",
        input_manifest={
            "part_geometry": "door_trim.step",
            "material": "PP-TD20",
            "wall_thickness": "2.5mm",
            "mold_concept": "two_plate",
        },
    )
    assert result["valid"] is True
    assert "gate_runner" in result["validation_checks"]
    assert "warpage" in result["validation_checks"]
    assert result["machine_start_allowed"] is False
    assert result["physical_execution_performed"] is False
    assert "openinjmoldsim" in [item["tool_id"] for item in result["tool_candidates"]]


def test_manufacturing_validation_requires_process_inputs_and_blocks_machine_start():
    with pytest.raises(ValueError, match="Missing required manufacturing inputs"):
        manufacturing.validate_process_request(
            "cnc_machining", input_manifest={"part_geometry": "bracket.step"}
        )
    with pytest.raises(PermissionError, match="does not start or control"):
        manufacturing.validate_process_request(
            "cnc_machining",
            input_manifest={
                "part_geometry": "bracket.step", "stock_geometry": "stock.step",
                "material": "Al6061", "machine_profile": "5_axis", "tooling": ["endmill"],
            },
            machine_start_requested=True,
        )


def test_process_preflight_checks_only_the_explicitly_selected_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    manifest = {
        "part_geometry": "door_trim.step", "material": "PP",
        "wall_thickness": "2.5mm", "mold_concept": "two_plate",
        "tool_id": "freecad",
    }
    with pytest.raises(PermissionError, match="not enabled"):
        manufacturing.validate_process_request(
            "injection_molding", input_manifest=manifest, preflight_tools=True
        )

    _enable_tool(tmp_path, monkeypatch, "freecad")
    result = manufacturing.validate_process_request(
        "injection_molding", input_manifest=manifest, preflight_tools=True
    )
    assert result["selected_tool"]["tool_id"] == "freecad"


def test_process_rejects_tool_not_in_its_candidate_list():
    with pytest.raises(ValueError, match="supported tool_id"):
        manufacturing.validate_process_request(
            "injection_molding",
            input_manifest={
                "part_geometry": "door_trim.step", "material": "PP",
                "wall_thickness": "2.5mm", "mold_concept": "two_plate",
                "tool_id": "linuxcnc",
            },
            preflight_tools=True,
        )
