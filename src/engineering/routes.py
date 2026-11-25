"""Authenticated API for the EVA Engineering Tool Registry.

These routes configure and inspect tools. They do not execute tools or start machines.
"""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

try:
    from ..rag.rag_routes_v2 import get_user_id
except ImportError:
    from rag.rag_routes_v2 import get_user_id
from . import job_manager, manufacturing, mcp_gateway, tool_registry, workflow_engine


router = APIRouter(prefix="/engineering", tags=["Engineering Suite"])


def _authenticated(user_id: str) -> str:
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


class ToolValidationRequest(BaseModel):
    executable_path: str
    version: Optional[str] = None


class ToolEnableRequest(BaseModel):
    enabled: bool
    approved: bool
    license_confirmed: bool = False


class ToolRegistrationRequest(BaseModel):
    tool_id: str
    name: str
    category: str
    integration_type: str
    license_type: str
    capabilities: list[str]
    notes: str = ""
    executable_candidates: list[str] = []


class EngineeringJobRequest(BaseModel):
    job_type: str
    tool_id: str
    workspace_id: str
    machine_id: str
    input_manifest: dict[str, Any] = Field(default_factory=dict)
    data_id: Optional[str] = None
    classification: str = "private"
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    requires_approval: bool = True


class JobApprovalRequest(BaseModel):
    approved: bool


class JobClaimRequest(BaseModel):
    machine_id: str


class JobCompletionRequest(BaseModel):
    output_manifest: dict[str, Any] = Field(default_factory=dict)


class JobFailureRequest(BaseModel):
    error: str


class WorkflowInstantiateRequest(BaseModel):
    workspace_id: str
    machine_id: str
    data_id: Optional[str] = None
    input_manifest: dict[str, Any] = Field(default_factory=dict)


class MCPToolRegistrationRequest(BaseModel):
    tool_name: str
    description: str = ""
    source_reviewed: bool
    approved: bool


class MCPToolEnableRequest(BaseModel):
    enabled: bool
    approved: bool


class MCPAuthorizationRequest(BaseModel):
    server_id: str
    tool_name: str
    workspace_id: str
    data_id: str
    action: str


class ManufacturingValidationRequest(BaseModel):
    input_manifest: dict[str, Any] = Field(default_factory=dict)
    machine_start_requested: bool = False
    preflight_tools: bool = False


@router.get("/tools")
def engineering_tools(
    category: Optional[str] = None,
    enabled_only: bool = False,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    return {"ok": True, "tools": tool_registry.list_tools(
        category=category, enabled_only=enabled_only
    )}


@router.get("/tools/{tool_id}")
def engineering_tool_detail(
    tool_id: str,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    tool = tool_registry.get_tool(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Engineering tool not found")
    return {"ok": True, "tool": tool}


@router.post("/tools")
def engineering_register_tool(
    req: ToolRegistrationRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        tool = tool_registry.register_tool(
            **req.dict(), actor_id=actor_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Tool registration failed: {type(exc).__name__}")
    return {"ok": True, "tool": tool}


@router.post("/tools/{tool_id}/validate")
def engineering_validate_tool(
    tool_id: str,
    req: ToolValidationRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        tool = tool_registry.validate_tool_installation(
            tool_id,
            executable_path=req.executable_path,
            actor_id=actor_id,
            version=req.version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering tool not found")
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "tool": tool, "execution_performed": False}


@router.post("/tools/{tool_id}/enable")
def engineering_enable_tool(
    tool_id: str,
    req: ToolEnableRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        tool = tool_registry.set_tool_enabled(
            tool_id,
            enabled=req.enabled,
            actor_id=actor_id,
            approved=req.approved,
            license_confirmed=req.license_confirmed,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering tool not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "ok": True,
        "tool": tool,
        "execution_performed": False,
        "machine_started": False,
    }


@router.post("/jobs")
def engineering_create_job(
    req: EngineeringJobRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        job = job_manager.create_job(
            **req.dict(), owner_id=actor_id, actor_id=actor_id
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "job": job, "execution_performed": False}


@router.get("/jobs")
def engineering_list_jobs(
    workspace_id: str,
    status: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    owner_id = _authenticated(user_id)
    return {
        "ok": True,
        "jobs": job_manager.list_jobs(
            owner_id=owner_id, workspace_id=workspace_id, status=status
        ),
    }


@router.get("/jobs/{job_id}")
def engineering_get_job(
    job_id: str,
    workspace_id: str,
    user_id: str = Depends(get_user_id),
):
    owner_id = _authenticated(user_id)
    job = job_manager.get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Engineering job not found")
    return {"ok": True, "job": job}


@router.post("/jobs/{job_id}/approve")
def engineering_approve_job(
    job_id: str,
    workspace_id: str,
    req: JobApprovalRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        job = job_manager.approve_job(
            job_id, owner_id=actor_id, workspace_id=workspace_id,
            actor_id=actor_id, approved=req.approved,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "job": job, "execution_performed": False}


@router.post("/jobs/{job_id}/claim")
def engineering_claim_job(
    job_id: str,
    workspace_id: str,
    req: JobClaimRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        job = job_manager.claim_job(
            job_id, owner_id=actor_id, workspace_id=workspace_id,
            machine_id=req.machine_id, actor_id=actor_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering job not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "ok": True, "job": job, "execution_performed": False,
        "machine_started": False, "machine_failover": False,
    }


@router.post("/jobs/{job_id}/complete")
def engineering_complete_job(
    job_id: str,
    workspace_id: str,
    req: JobCompletionRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        job = job_manager.complete_job(
            job_id, owner_id=actor_id, workspace_id=workspace_id,
            actor_id=actor_id, output_manifest=req.output_manifest,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "job": job, "execution_performed": False}


@router.post("/jobs/{job_id}/fail")
def engineering_fail_job(
    job_id: str,
    workspace_id: str,
    req: JobFailureRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        job = job_manager.fail_job(
            job_id, owner_id=actor_id, workspace_id=workspace_id,
            actor_id=actor_id, error=req.error,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "job": job, "execution_performed": False}


@router.get("/workflows")
def engineering_workflows(user_id: str = Depends(get_user_id)):
    _authenticated(user_id)
    return {"ok": True, "workflows": workflow_engine.list_workflows()}


@router.get("/workflows/{workflow_id}")
def engineering_workflow_detail(
    workflow_id: str,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    try:
        return {"ok": True, "workflow": workflow_engine.get_workflow(workflow_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering workflow not found")


@router.post("/workflows/{workflow_id}/validate")
def engineering_validate_workflow(
    workflow_id: str,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    try:
        result = workflow_engine.validate_workflow(workflow_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering workflow not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "validation": result, "execution_performed": False}


@router.post("/workflows/{workflow_id}/instantiate")
def engineering_instantiate_workflow(
    workflow_id: str,
    req: WorkflowInstantiateRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        result = workflow_engine.instantiate_workflow(
            workflow_id,
            owner_id=actor_id,
            workspace_id=req.workspace_id,
            machine_id=req.machine_id,
            actor_id=actor_id,
            data_id=req.data_id,
            input_manifest=req.input_manifest,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Engineering workflow or tool not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "workflow": result, "execution_performed": False}


@router.get("/mcp/tools")
def engineering_mcp_tools(
    server_id: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    return {"ok": True, "tools": mcp_gateway.list_mcp_tools(server_id)}


@router.post("/mcp/servers/{server_id}/tools")
def engineering_register_mcp_tool(
    server_id: str,
    req: MCPToolRegistrationRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        item = mcp_gateway.register_mcp_tool(
            server_id, req.tool_name, actor_id=actor_id,
            description=req.description,
            source_reviewed=req.source_reviewed,
            approved=req.approved,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="MCP server not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "tool": item, "execution_performed": False}


@router.post("/mcp/servers/{server_id}/tools/{tool_name}/enable")
def engineering_enable_mcp_tool(
    server_id: str,
    tool_name: str,
    req: MCPToolEnableRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        item = mcp_gateway.set_mcp_tool_enabled(
            server_id, tool_name, enabled=req.enabled,
            actor_id=actor_id, approved=req.approved,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="MCP tool not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "tool": item, "execution_performed": False}


@router.post("/mcp/authorize")
def engineering_authorize_mcp_call(
    req: MCPAuthorizationRequest,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    try:
        result = mcp_gateway.authorize_mcp_call(
            req.server_id, req.tool_name, actor_id=actor_id,
            workspace_id=req.workspace_id, data_id=req.data_id, action=req.action,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="MCP tool or governed data not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "authorization": result, "execution_performed": False}


@router.get("/mcp/audit")
def engineering_mcp_audit(
    workspace_id: str,
    user_id: str = Depends(get_user_id),
):
    actor_id = _authenticated(user_id)
    return {
        "ok": True,
        "events": mcp_gateway.list_mcp_audit(
            actor_id=actor_id, workspace_id=workspace_id
        ),
    }


@router.get("/manufacturing/processes")
def engineering_manufacturing_processes(
    category: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    return {"ok": True, "processes": manufacturing.list_processes(category)}


@router.get("/manufacturing/processes/{process_id}")
def engineering_manufacturing_process_detail(
    process_id: str,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    try:
        return {"ok": True, "process": manufacturing.get_process(process_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Manufacturing process not found")


@router.post("/manufacturing/processes/{process_id}/validate")
def engineering_validate_manufacturing_process(
    process_id: str,
    req: ManufacturingValidationRequest,
    user_id: str = Depends(get_user_id),
):
    _authenticated(user_id)
    try:
        result = manufacturing.validate_process_request(
            process_id,
            input_manifest=req.input_manifest,
            machine_start_requested=req.machine_start_requested,
            preflight_tools=req.preflight_tools,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Manufacturing process not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "validation": result, "physical_execution_performed": False}
