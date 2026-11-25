from pathlib import Path

import pytest

from src.engineering import job_manager, tool_registry


def _enable_tool(tmp_path, monkeypatch, tool_id="freecad"):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / f"{tool_id}-stub.exe"
    executable.write_text("local executable placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        tool_id, executable_path=str(executable), actor_id="owner-1"
    )
    tool_registry.set_tool_enabled(
        tool_id, enabled=True, actor_id="owner-1", approved=True,
        license_confirmed=(tool_id == "catia"),
    )


def test_engineering_job_requires_enabled_validated_tool_and_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    with pytest.raises(PermissionError, match="not enabled"):
        job_manager.create_job(
            job_type="cad_modify", tool_id="freecad", owner_id="owner-1",
            workspace_id="cad", machine_id="machine-b", actor_id="owner-1",
            input_manifest={"part": "door-trim"},
        )

    _enable_tool(tmp_path, monkeypatch)
    job = job_manager.create_job(
        job_type="cad_modify", tool_id="freecad", owner_id="owner-1",
        workspace_id="cad", machine_id="machine-b", actor_id="owner-1",
        input_manifest={"part": "door-trim", "operation": "add_rib"},
        requires_approval=False,
    )
    assert job["status"] == "pending_approval"
    assert job["approval_required"] is True

    with pytest.raises(ValueError, match="cannot be claimed"):
        job_manager.claim_job(
            job["job_id"], owner_id="owner-1", workspace_id="cad",
            machine_id="machine-b", actor_id="worker-1",
        )

    approved = job_manager.approve_job(
        job["job_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="reviewer-1", approved=True,
    )
    assert approved["status"] == "queued"
    assert approved["approved_by"] == "reviewer-1"

    claimed = job_manager.claim_job(
        job["job_id"], owner_id="owner-1", workspace_id="cad",
        machine_id="machine-b", actor_id="worker-1",
    )
    assert claimed["status"] == "running"
    completed = job_manager.complete_job(
        job["job_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="worker-1", output_manifest={"result": "model_updated"},
    )
    assert completed["status"] == "succeeded"
    assert completed["output_manifest"]["result"] == "model_updated"


def test_job_is_bound_to_selected_machine_no_automatic_failover(tmp_path, monkeypatch):
    _enable_tool(tmp_path, monkeypatch)
    job = job_manager.create_job(
        job_type="fea_solve", tool_id="freecad", owner_id="owner-1",
        workspace_id="cad", machine_id="machine-a", actor_id="owner-1",
        input_manifest={"model": "bracket.step"}, requires_approval=False,
    )
    with pytest.raises(PermissionError, match="automatic machine failover is disabled"):
        job_manager.claim_job(
            job["job_id"], owner_id="owner-1", workspace_id="cad",
            machine_id="machine-b", actor_id="worker-1",
        )
    assert job_manager.get_job(
        job["job_id"], owner_id="owner-1", workspace_id="cad"
    )["status"] == "queued"


def test_machine_start_jobs_are_rejected_and_machine_operations_need_approval(tmp_path, monkeypatch):
    _enable_tool(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="does not automatically start CNC machines"):
        job_manager.create_job(
            job_type="machine_start", tool_id="freecad", owner_id="owner-1",
            workspace_id="cad", machine_id="cnc-1", actor_id="owner-1",
            input_manifest={"program": "part.nc"},
        )

    job = job_manager.create_job(
        job_type="machine_operation", tool_id="freecad", owner_id="owner-1",
        workspace_id="cad", machine_id="cnc-1", actor_id="owner-1",
        input_manifest={"program": "part.nc"}, requires_approval=False,
    )
    assert job["status"] == "pending_approval"


def test_job_events_are_append_only_and_scope_isolation_is_enforced(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "engineering.db"
    _enable_tool(tmp_path, monkeypatch)
    job = job_manager.create_job(
        job_type="analysis", tool_id="freecad", owner_id="owner-1",
        workspace_id="cad", machine_id="machine-b", actor_id="owner-1",
        input_manifest={"analysis": "geometry_check"}, requires_approval=False,
    )
    assert job_manager.get_job(
        job["job_id"], owner_id="other-user", workspace_id="cad"
    ) is None
    events = job_manager.list_job_events(
        job["job_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert events[0]["action"] == "job_created"

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE engineering_job_events SET action='tampered' WHERE job_id=?",
                (job["job_id"],),
            )
    finally:
        conn.close()


def test_job_creation_requires_execute_permission_for_governed_data(tmp_path, monkeypatch):
    from src.data_governance import register_import, set_permissions

    _enable_tool(tmp_path, monkeypatch)
    source = tmp_path / "governed.txt"
    source.write_text("governed engineering input", encoding="utf-8")
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
    with pytest.raises(PermissionError, match="Execute permission"):
        job_manager.create_job(
            job_type="fea_solve", tool_id="freecad", owner_id="reviewer-1",
            workspace_id="cad", machine_id="machine-b", actor_id="reviewer-1",
            data_id=record["data_id"], input_manifest={"data_id": record["data_id"]},
            requires_approval=False,
        )


def test_engineering_job_api_is_authenticated_and_never_executes_tools(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    _enable_tool(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as error:
        routes.engineering_list_jobs(workspace_id="cad", user_id="anonymous")
    assert error.value.status_code == 401

    created = routes.engineering_create_job(
        routes.EngineeringJobRequest(
            job_type="cad_modify", tool_id="freecad", workspace_id="cad",
            machine_id="machine-b", input_manifest={"operation": "create_boss"},
        ),
        user_id="owner-1",
    )
    job_id = created["job"]["job_id"]
    assert created["job"]["status"] == "pending_approval"
    assert created["execution_performed"] is False

    approved = routes.engineering_approve_job(
        job_id, workspace_id="cad",
        req=routes.JobApprovalRequest(approved=True), user_id="owner-1",
    )
    assert approved["job"]["status"] == "queued"

    with pytest.raises(HTTPException) as wrong_machine:
        routes.engineering_claim_job(
            job_id, workspace_id="cad",
            req=routes.JobClaimRequest(machine_id="machine-a"), user_id="owner-1",
        )
    assert wrong_machine.value.status_code == 403

    claimed = routes.engineering_claim_job(
        job_id, workspace_id="cad",
        req=routes.JobClaimRequest(machine_id="machine-b"), user_id="owner-1",
    )
    assert claimed["job"]["status"] == "running"
    assert claimed["execution_performed"] is False
    assert claimed["machine_started"] is False
    assert claimed["machine_failover"] is False



def test_job_creation_cannot_assign_another_user_as_owner(tmp_path, monkeypatch):
    _enable_tool(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="authenticated owner"):
        job_manager.create_job(
            job_type="analysis", tool_id="freecad", owner_id="owner-1",
            workspace_id="cad", machine_id="machine-b", actor_id="attacker",
            input_manifest={"operation": "geometry_check"}, requires_approval=False,
        )
    assert job_manager.list_jobs(owner_id="owner-1", workspace_id="cad") == []
