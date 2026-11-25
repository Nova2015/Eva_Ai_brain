from pathlib import Path

import pytest

from src.data_governance import list_audit_events, register_import
from src.engineering import gmsh_adapter


def _register_step(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    step = tmp_path / "part.step"
    step.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('fixture'),'2;1');\nENDSEC;\n"
        "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    record = register_import(
        step, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1",
    )
    return step, record


def test_gmsh_script_requires_governed_step_and_creates_reviewable_script(tmp_path, monkeypatch):
    _, record = _register_step(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    manifest = gmsh_adapter.build_gmsh_mesh_script(
        data_id=record["data_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", mesh_size_min_mm=0.5, mesh_size_max_mm=2.0,
        algorithm_3d=10, element_order=1, approved=True,
    )
    script = Path(manifest["script_path"]).read_text(encoding="utf-8")
    assert manifest["status"] == "prepared_for_review"
    assert manifest["mesh_generation_performed"] is False
    assert manifest["source_data_id"] == record["data_id"]
    assert "SetFactory(" in script
    assert "Merge " in script
    assert "Mesh.MshFileVersion = 2.2;" in script
    assert "Mesh 3;" in script
    assert manifest["mesh_output_path"].endswith(".msh")
    events = list_audit_events(record["data_id"], owner_id="owner-1", workspace_id="cad")
    assert events[-1]["action"] == "gmsh_mesh_script_prepared"


def test_gmsh_script_rejects_missing_approval_and_execute_permission(tmp_path, monkeypatch):
    _, record = _register_step(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    args = dict(
        data_id=record["data_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", mesh_size_min_mm=0.5, mesh_size_max_mm=2.0,
        approved=True,
    )
    with pytest.raises(PermissionError, match="Explicit approval"):
        gmsh_adapter.build_gmsh_mesh_script(**dict(args, approved=False))

    with pytest.raises(PermissionError, match="Execute permission"):
        gmsh_adapter.build_gmsh_mesh_script(**dict(args, workspace_id="other-workspace"))


def test_gmsh_script_rejects_changed_source_hash_and_bad_mesh_settings(tmp_path, monkeypatch):
    step, record = _register_step(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    args = dict(
        data_id=record["data_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", mesh_size_min_mm=0.5, mesh_size_max_mm=2.0,
        approved=True,
    )
    step.write_text(step.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        gmsh_adapter.build_gmsh_mesh_script(**args)

    step, record = _register_step(tmp_path / "second", monkeypatch)
    with pytest.raises(ValueError, match="cannot exceed"):
        gmsh_adapter.build_gmsh_mesh_script(
            **dict(args, data_id=record["data_id"], mesh_size_min_mm=3, mesh_size_max_mm=2)
        )
    with pytest.raises(ValueError, match="Unsupported 3D mesh algorithm"):
        gmsh_adapter.build_gmsh_mesh_script(
            **dict(args, data_id=record["data_id"], algorithm_3d=10.0)
        )


def test_gmsh_mesh_script_api_is_authenticated_and_never_runs_gmsh(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    _, record = _register_step(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    request = routes.GmshMeshScriptRequest(
        data_id=record["data_id"], workspace_id="cad",
        mesh_size_min_mm=0.5, mesh_size_max_mm=2.0, approved=True,
    )
    with pytest.raises(HTTPException) as error:
        routes.engineering_build_gmsh_mesh_script(request, user_id="anonymous")
    assert error.value.status_code == 401

    result = routes.engineering_build_gmsh_mesh_script(request, user_id="owner-1")
    assert result["mesh_script"]["status"] == "prepared_for_review"
    assert result["mesh_generation_performed"] is False
    assert result["solver_execution_performed"] is False
