import pytest

from src.engineering import job_manager, tool_registry, workflow_engine


def _enable_freecad(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    tool_registry.list_tools()
    executable = tmp_path / "freecad-stub.exe"
    executable.write_text("placeholder", encoding="utf-8")
    tool_registry.validate_tool_installation(
        "freecad", executable_path=str(executable), actor_id="owner-1"
    )
    tool_registry.set_tool_enabled(
        "freecad", enabled=True, actor_id="owner-1", approved=True
    )


def test_all_six_engineering_workflow_templates_are_valid_dags():
    templates = workflow_engine.list_workflows()
    assert len(templates) == 6
    expected = {
        "automotive_door_trim", "plastic_bracket", "injection_mold",
        "cnc_component", "sheet_metal_bracket", "plastic_assembly",
    }
    assert {item["workflow_id"] for item in templates} == expected

    for workflow_id in expected:
        result = workflow_engine.validate_workflow(workflow_id)
        assert result["valid"] is True
        assert len(result["step_order"]) == templates[
            next(i for i, item in enumerate(templates) if item["workflow_id"] == workflow_id)
        ]["step_count"]


def test_workflow_validation_rejects_cycles_and_unknown_dependencies(monkeypatch):
    monkeypatch.setitem(
        workflow_engine._WORKFLOWS,
        "cyclic_test",
        {
            "name": "Cycle",
            "description": "invalid",
            "steps": [
                {"step_id": "a", "tool_id": "freecad", "job_type": "analysis",
                 "depends_on": ["b"], "requires_approval": False},
                {"step_id": "b", "tool_id": "freecad", "job_type": "analysis",
                 "depends_on": ["a"], "requires_approval": False},
            ],
        },
    )
    with pytest.raises(ValueError, match="cycle"):
        workflow_engine.validate_workflow("cyclic_test")

    monkeypatch.setitem(
        workflow_engine._WORKFLOWS,
        "unknown_dependency_test",
        {
            "name": "Unknown",
            "description": "invalid",
            "steps": [
                {"step_id": "a", "tool_id": "freecad", "job_type": "analysis",
                 "depends_on": ["missing"], "requires_approval": False},
            ],
        },
    )
    with pytest.raises(ValueError, match="unknown dependency"):
        workflow_engine.validate_workflow("unknown_dependency_test")


def test_workflow_instantiation_preflights_tools_before_creating_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    with pytest.raises(PermissionError, match="not enabled"):
        workflow_engine.instantiate_workflow(
            "plastic_bracket", owner_id="owner-1", workspace_id="cad",
            machine_id="machine-b", actor_id="owner-1", data_id=None,
            input_manifest={"part": "bracket"},
        )
    assert job_manager.list_jobs(owner_id="owner-1", workspace_id="cad") == []


def test_workflow_jobs_wait_for_dependencies_to_succeed(tmp_path, monkeypatch):
    _enable_freecad(tmp_path, monkeypatch)
    monkeypatch.setitem(
        workflow_engine._WORKFLOWS,
        "two_step_test",
        {
            "name": "Two Step Test",
            "description": "dependency test",
            "steps": [
                {"step_id": "first", "tool_id": "freecad", "job_type": "analysis",
                 "depends_on": [], "requires_approval": False},
                {"step_id": "second", "tool_id": "freecad", "job_type": "analysis",
                 "depends_on": ["first"], "requires_approval": False},
            ],
        },
    )

    workflow = workflow_engine.instantiate_workflow(
        "two_step_test", owner_id="owner-1", workspace_id="cad",
        machine_id="machine-b", actor_id="owner-1", data_id=None,
        input_manifest={"part": "bracket"},
    )
    assert workflow["execution_performed"] is False
    jobs = {job["step_id"]: job for job in workflow["jobs"]}
    assert jobs["first"]["status"] == "queued"
    assert jobs["second"]["depends_on"] == ["first"]

    with pytest.raises(ValueError, match="dependencies are not complete"):
        job_manager.claim_job(
            jobs["second"]["job_id"], owner_id="owner-1", workspace_id="cad",
            machine_id="machine-b", actor_id="worker-1",
        )

    job_manager.claim_job(
        jobs["first"]["job_id"], owner_id="owner-1", workspace_id="cad",
        machine_id="machine-b", actor_id="worker-1",
    )
    job_manager.complete_job(
        jobs["first"]["job_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="worker-1", output_manifest={"result": "ok"},
    )
    claimed = job_manager.claim_job(
        jobs["second"]["job_id"], owner_id="owner-1", workspace_id="cad",
        machine_id="machine-b", actor_id="worker-1",
    )
    assert claimed["status"] == "running"


def test_workflow_instantiation_requires_explicit_machine_and_no_auto_start(tmp_path, monkeypatch):
    _enable_freecad(tmp_path, monkeypatch)
    monkeypatch.setitem(
        workflow_engine._WORKFLOWS,
        "machine_start_test",
        {
            "name": "Unsafe",
            "description": "must reject",
            "steps": [
                {"step_id": "start", "tool_id": "freecad", "job_type": "machine_start",
                 "depends_on": [], "requires_approval": True},
            ],
        },
    )
    with pytest.raises(ValueError, match="automatic machine start"):
        workflow_engine.validate_workflow("machine_start_test")
