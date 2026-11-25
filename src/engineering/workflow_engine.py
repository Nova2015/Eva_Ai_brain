"""EVA Engineering workflow templates and dependency validation.

Templates describe engineering stages; they do not claim that any tool is installed,
validated, or enabled. Instantiation is blocked until every required tool is approved.
"""
from __future__ import annotations

import uuid
from typing import Any

from . import job_manager, tool_registry


_WORKFLOWS: dict[str, dict[str, Any]] = {
    "automotive_door_trim": {
        "name": "Automotive Door Trim Engineering",
        "description": "Governed CAD, DFM, mold, mesh, FEA, mold-flow and visualization workflow.",
        "steps": [
            {"step_id": "cad_model", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "dfm_review", "tool_id": "freecad", "job_type": "analysis", "depends_on": ["cad_model"], "requires_approval": True},
            {"step_id": "mold_design", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": ["dfm_review"], "requires_approval": True},
            {"step_id": "mesh_generation", "tool_id": "gmsh", "job_type": "mesh_generate", "depends_on": ["mold_design"], "requires_approval": True},
            {"step_id": "structural_fea", "tool_id": "calculix", "job_type": "fea_solve", "depends_on": ["mesh_generation"], "requires_approval": True},
            {"step_id": "mold_flow", "tool_id": "openinjmoldsim", "job_type": "mold_flow", "depends_on": ["mold_design"], "requires_approval": True},
            {"step_id": "visualization", "tool_id": "blender", "job_type": "visualization", "depends_on": ["structural_fea", "mold_flow"], "requires_approval": False},
            {"step_id": "engineering_report", "tool_id": "python_reporting", "job_type": "report_generate", "depends_on": ["visualization"], "requires_approval": True},
        ],
    },
    "plastic_bracket": {
        "name": "Plastic Bracket Engineering",
        "description": "CAD, material/DFM review, meshing, structural analysis and visualization.",
        "steps": [
            {"step_id": "cad_model", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "mesh_generation", "tool_id": "gmsh", "job_type": "mesh_generate", "depends_on": ["cad_model"], "requires_approval": True},
            {"step_id": "structural_fea", "tool_id": "calculix", "job_type": "fea_solve", "depends_on": ["mesh_generation"], "requires_approval": True},
            {"step_id": "visualization", "tool_id": "blender", "job_type": "visualization", "depends_on": ["structural_fea"], "requires_approval": False},
            {"step_id": "engineering_report", "tool_id": "python_reporting", "job_type": "report_generate", "depends_on": ["visualization"], "requires_approval": True},
        ],
    },
    "injection_mold": {
        "name": "Injection Mold Engineering",
        "description": "Core/cavity design, mold-flow, structural validation and CAM preparation.",
        "steps": [
            {"step_id": "mold_cad", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "mold_flow", "tool_id": "openinjmoldsim", "job_type": "mold_flow", "depends_on": ["mold_cad"], "requires_approval": True},
            {"step_id": "mold_mesh", "tool_id": "gmsh", "job_type": "mesh_generate", "depends_on": ["mold_cad"], "requires_approval": True},
            {"step_id": "mold_fea", "tool_id": "calculix", "job_type": "fea_solve", "depends_on": ["mold_mesh"], "requires_approval": True},
            {"step_id": "cam_preparation", "tool_id": "freecad_path", "job_type": "cam_postprocess", "depends_on": ["mold_fea", "mold_flow"], "requires_approval": True},
            {"step_id": "manufacturing_report", "tool_id": "python_reporting", "job_type": "report_generate", "depends_on": ["cam_preparation"], "requires_approval": True},
        ],
    },
    "cnc_component": {
        "name": "CNC Component Engineering",
        "description": "CAD, CAM, toolpath validation and operator approval; no machine start.",
        "steps": [
            {"step_id": "component_cad", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "cam_toolpath", "tool_id": "freecad_path", "job_type": "cam_postprocess", "depends_on": ["component_cad"], "requires_approval": True},
            {"step_id": "gcode_validation", "tool_id": "gcode_validator", "job_type": "gcode_generate", "depends_on": ["cam_toolpath"], "requires_approval": True},
            {"step_id": "operator_review", "tool_id": "gcode_validator", "job_type": "machine_operation", "depends_on": ["gcode_validation"], "requires_approval": True},
        ],
    },
    "sheet_metal_bracket": {
        "name": "Sheet Metal Bracket Engineering",
        "description": "Parametric sheet-metal design, forming analysis, CAM and inspection preparation.",
        "steps": [
            {"step_id": "sheet_metal_cad", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "forming_mesh", "tool_id": "gmsh", "job_type": "mesh_generate", "depends_on": ["sheet_metal_cad"], "requires_approval": True},
            {"step_id": "forming_analysis", "tool_id": "code_aster", "job_type": "fea_solve", "depends_on": ["forming_mesh"], "requires_approval": True},
            {"step_id": "cam_preparation", "tool_id": "freecad_path", "job_type": "cam_postprocess", "depends_on": ["forming_analysis"], "requires_approval": True},
            {"step_id": "inspection_report", "tool_id": "python_reporting", "job_type": "report_generate", "depends_on": ["cam_preparation"], "requires_approval": True},
        ],
    },
    "plastic_assembly": {
        "name": "Plastic Assembly Engineering",
        "description": "Assembly design, fit/tolerance checks, structural validation and exploded visualization.",
        "steps": [
            {"step_id": "assembly_cad", "tool_id": "freecad", "job_type": "cad_modify", "depends_on": [], "requires_approval": True},
            {"step_id": "fitment_mesh", "tool_id": "gmsh", "job_type": "mesh_generate", "depends_on": ["assembly_cad"], "requires_approval": True},
            {"step_id": "assembly_fea", "tool_id": "calculix", "job_type": "fea_solve", "depends_on": ["fitment_mesh"], "requires_approval": True},
            {"step_id": "exploded_visualization", "tool_id": "blender", "job_type": "visualization", "depends_on": ["assembly_fea"], "requires_approval": False},
            {"step_id": "assembly_report", "tool_id": "python_reporting", "job_type": "report_generate", "depends_on": ["exploded_visualization"], "requires_approval": True},
        ],
    },
}


def list_workflows() -> list[dict[str, Any]]:
    return [
        {
            "workflow_id": workflow_id,
            "name": template["name"],
            "description": template["description"],
            "step_count": len(template["steps"]),
        }
        for workflow_id, template in sorted(_WORKFLOWS.items())
    ]


def get_workflow(workflow_id: str) -> dict[str, Any]:
    template = _WORKFLOWS.get(workflow_id)
    if template is None:
        raise KeyError(f"Unknown engineering workflow: {workflow_id}")
    return {
        "workflow_id": workflow_id,
        "name": template["name"],
        "description": template["description"],
        "steps": [dict(step) for step in template["steps"]],
    }


def validate_workflow(workflow_id: str) -> dict[str, Any]:
    template = get_workflow(workflow_id)
    steps = template["steps"]
    by_id = {step["step_id"]: step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("Workflow contains duplicate step IDs")

    for step in steps:
        if step["job_type"] == "machine_start":
            raise ValueError("Workflow templates cannot contain automatic machine start")
        for dependency in step["depends_on"]:
            if dependency not in by_id:
                raise ValueError(
                    f"Workflow step {step['step_id']} has unknown dependency {dependency}"
                )

    ordered: list[dict[str, Any]] = []
    pending = {step["step_id"]: step for step in steps}
    completed: set[str] = set()
    while pending:
        ready = [
            step for step in pending.values()
            if set(step["depends_on"]).issubset(completed)
        ]
        if not ready:
            raise ValueError("Workflow dependency graph contains a cycle")
        for step in ready:
            ordered.append(step)
            completed.add(step["step_id"])
            pending.pop(step["step_id"])

    return {
        "workflow_id": workflow_id,
        "valid": True,
        "step_order": [step["step_id"] for step in ordered],
        "steps": ordered,
    }


def instantiate_workflow(
    workflow_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    machine_id: str,
    actor_id: str,
    data_id: Optional[str],
    input_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Create approval-gated jobs in topological order; no tool is executed."""
    validated = validate_workflow(workflow_id)
    ordered_steps = validated["steps"]

    # Preflight all tool gates before creating any partial workflow.
    for step in ordered_steps:
        tool_registry.resolve_tool(step["tool_id"])

    workflow_run_id = "EVA-WORKFLOW-" + uuid.uuid4().hex.upper()
    created_jobs: list[dict[str, Any]] = []
    try:
        for step in ordered_steps:
            job = job_manager.create_job(
                job_type=step["job_type"],
                tool_id=step["tool_id"],
                owner_id=owner_id,
                workspace_id=workspace_id,
                machine_id=machine_id,
                actor_id=actor_id,
                input_manifest={
                    **input_manifest,
                    "workflow_id": workflow_id,
                    "workflow_step": step["step_id"],
                },
                data_id=data_id,
                workflow_id=workflow_run_id,
                step_id=step["step_id"],
                depends_on=step["depends_on"],
                requires_approval=step["requires_approval"],
            )
            created_jobs.append(job)
    except Exception as exc:
        for job in created_jobs:
            try:
                job_manager.fail_job(
                    job["job_id"], owner_id=owner_id, workspace_id=workspace_id,
                    actor_id=actor_id, error=f"Workflow creation rolled back: {type(exc).__name__}",
                )
            except Exception:
                pass
        raise

    return {
        "workflow_id": workflow_id,
        "workflow_run_id": workflow_run_id,
        "name": get_workflow(workflow_id)["name"],
        "jobs": created_jobs,
        "execution_performed": False,
    }
