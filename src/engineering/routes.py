"""Authenticated API for the EVA Engineering Tool Registry.

These routes configure and inspect tools. They do not execute tools or start machines.
"""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..rag.rag_routes_v2 import get_user_id
from . import tool_registry


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
            **req.model_dump(), actor_id=actor_id
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
