"""Local engineering job queue with approval and exact-machine binding.

This module manages job state only. It does not invoke executables, send MCP calls,
start CNC machines, or fail over to another machine.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import tool_registry


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "eva_engineering.db"

_ALWAYS_APPROVAL_JOB_TYPES = frozenset({
    "cad_modify", "cad_delete", "file_overwrite", "file_delete",
    "cam_postprocess", "gcode_generate", "machine_operation",
    "machine_start", "training_export", "tool_install",
})


def _db_path() -> Path:
    return Path(os.getenv("EVA_ENGINEERING_DB", str(DEFAULT_DB_PATH))).expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_actor(actor_id: str) -> None:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _initialize(conn)
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS engineering_jobs (
            job_id TEXT PRIMARY KEY,
            workflow_id TEXT,
            step_id TEXT,
            job_type TEXT NOT NULL,
            tool_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            data_id TEXT,
            classification TEXT NOT NULL,
            input_manifest_json TEXT NOT NULL,
            depends_on_json TEXT NOT NULL DEFAULT '[]',
            output_manifest_json TEXT,
            status TEXT NOT NULL,
            approval_required INTEGER NOT NULL,
            approved_by TEXT,
            approved_at TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_engineering_jobs_scope
            ON engineering_jobs(owner_id, workspace_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_engineering_jobs_status
            ON engineering_jobs(status, machine_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_engineering_workflow_step
            ON engineering_jobs(workflow_id, step_id)
            WHERE workflow_id IS NOT NULL AND step_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS engineering_job_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TRIGGER IF NOT EXISTS trg_engineering_job_events_no_update
        BEFORE UPDATE ON engineering_job_events
        BEGIN
            SELECT RAISE(ABORT, 'engineering_job_events is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_engineering_job_events_no_delete
        BEFORE DELETE ON engineering_job_events
        BEGIN
            SELECT RAISE(ABORT, 'engineering_job_events is append-only');
        END;
        """
    )
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(engineering_jobs)").fetchall()
    }
    if "depends_on_json" not in columns:
        conn.execute(
            "ALTER TABLE engineering_jobs ADD COLUMN depends_on_json TEXT NOT NULL DEFAULT '[]'"
        )
    conn.commit()


def _event(
    conn: sqlite3.Connection,
    job_id: str,
    action: str,
    actor_id: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    conn.execute(
        """INSERT INTO engineering_job_events
           (job_id, action, actor_id, details_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (job_id, action, actor_id, json.dumps(details or {}, sort_keys=True), _now()),
    )


def _job_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    item = dict(row)
    item["input_manifest"] = json.loads(item.pop("input_manifest_json"))
    item["depends_on"] = json.loads(item.pop("depends_on_json") or "[]")
    raw_output = item.pop("output_manifest_json")
    item["output_manifest"] = json.loads(raw_output) if raw_output else None
    item["approval_required"] = bool(item["approval_required"])
    return item


def create_job(
    *,
    job_type: str,
    tool_id: str,
    owner_id: str,
    workspace_id: str,
    machine_id: str,
    actor_id: str,
    input_manifest: dict[str, Any],
    data_id: Optional[str] = None,
    classification: str = "private",
    workflow_id: Optional[str] = None,
    step_id: Optional[str] = None,
    requires_approval: bool = True,
    depends_on: Optional[list[str]] = None,
) -> dict[str, Any]:
    for field, value in (
        ("job_type", job_type), ("tool_id", tool_id), ("owner_id", owner_id),
        ("workspace_id", workspace_id), ("machine_id", machine_id), ("actor_id", actor_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if job_type == "machine_start":
        raise PermissionError("EVA does not automatically start CNC machines")
    if not isinstance(input_manifest, dict):
        raise ValueError("input_manifest must be a dictionary")
    dependencies = depends_on or []
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or not item.strip() for item in dependencies
    ):
        raise ValueError("depends_on must be a list of non-empty step IDs")
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("depends_on cannot contain duplicate step IDs")
    if step_id and not workflow_id:
        raise ValueError("step_id requires workflow_id")
    if dependencies and not workflow_id:
        raise ValueError("dependent jobs must belong to a workflow")
    if step_id and step_id in dependencies:
        raise ValueError("a workflow step cannot depend on itself")
    if not isinstance(requires_approval, bool):
        raise ValueError("requires_approval must be a boolean")
    if classification not in {
        "public", "private", "sensitive", "ip",
        "internal", "confidential", "restricted",
    }:
        raise ValueError("unsupported data classification")
    if data_id is not None and (not isinstance(data_id, str) or not data_id.strip()):
        raise ValueError("data_id must be a non-empty string when provided")
    try:
        manifest_json = json.dumps(input_manifest, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_manifest must be JSON serializable") from exc

    tool = tool_registry.resolve_tool(tool_id)
    if tool["machine_start_allowed"]:
        raise PermissionError("Tool is not permitted to start a machine")
    if data_id:
        try:
            from ..data_governance import authorize_data_action
        except ImportError:
            from data_governance import authorize_data_action
        if not authorize_data_action(
            data_id, actor_id=actor_id, workspace_id=workspace_id, action="execute"
        ):
            raise PermissionError("Execute permission is required for the governed source")

    approval_required = bool(requires_approval or job_type in _ALWAYS_APPROVAL_JOB_TYPES)
    job_id = "EVA-JOB-" + uuid.uuid4().hex.upper()
    status = "pending_approval" if approval_required else "queued"
    created_at = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO engineering_jobs
               (job_id, workflow_id, step_id, job_type, tool_id, owner_id,
                workspace_id, machine_id, data_id, classification,
                input_manifest_json, depends_on_json, status, approval_required, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, workflow_id, step_id, job_type, tool_id, owner_id,
                workspace_id, machine_id, data_id, classification,
                manifest_json, json.dumps(dependencies), status, int(approval_required), created_at,
            ),
        )
        _event(conn, job_id, "job_created", actor_id, {
            "status": status, "machine_id": machine_id, "tool_id": tool_id,
            "approval_required": approval_required,
        })
    return get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def approve_job(
    job_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    approved: bool,
) -> dict[str, Any]:
    if not approved:
        raise ValueError("Explicit job approval is required")
    _require_actor(actor_id)
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Engineering job not found: {job_id}")
        if row["status"] != "pending_approval":
            raise ValueError(f"Job cannot be approved from status {row['status']!r}")
        cursor = conn.execute(
            """UPDATE engineering_jobs
               SET status='queued', approved_by=?, approved_at=?
               WHERE job_id=? AND owner_id=? AND workspace_id=? AND status='pending_approval'""",
            (actor_id, _now(), job_id, owner_id, workspace_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Job approval was already processed")
        _event(conn, job_id, "job_approved", actor_id, {"approved": True})
    return get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def claim_job(
    job_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    machine_id: str,
    actor_id: str,
) -> dict[str, Any]:
    if not machine_id or not machine_id.strip():
        raise ValueError("machine_id is required")
    _require_actor(actor_id)
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Engineering job not found: {job_id}")
        if row["machine_id"] != machine_id:
            raise PermissionError(
                "Job is bound to its selected machine; automatic machine failover is disabled"
            )
        if row["status"] != "queued":
            raise ValueError(f"Job cannot be claimed from status {row['status']!r}")
        dependencies = json.loads(row["depends_on_json"] or "[]")
        if dependencies:
            if not row["workflow_id"]:
                raise ValueError("Dependent jobs must belong to a workflow")
            completed_steps = {
                item["step_id"]: item["status"]
                for item in conn.execute(
                    """SELECT step_id, status FROM engineering_jobs
                       WHERE workflow_id=? AND owner_id=? AND workspace_id=?""",
                    (row["workflow_id"], owner_id, workspace_id),
                ).fetchall()
            }
            blocked = [
                step_id for step_id in dependencies
                if completed_steps.get(step_id) != "succeeded"
            ]
            if blocked:
                raise ValueError(
                    "Workflow dependencies are not complete: " + ", ".join(blocked)
                )
        tool = tool_registry.resolve_tool(row["tool_id"])
        if tool["machine_start_allowed"]:
            raise PermissionError("Automatic machine start is prohibited")
        cursor = conn.execute(
            """UPDATE engineering_jobs SET status='running', started_at=?
               WHERE job_id=? AND owner_id=? AND workspace_id=? AND status='queued'""",
            (_now(), job_id, owner_id, workspace_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Job was already claimed by another worker")
        _event(conn, job_id, "job_claimed", actor_id, {"machine_id": machine_id})
    return get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def complete_job(
    job_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    output_manifest: dict[str, Any],
) -> dict[str, Any]:
    _require_actor(actor_id)
    if not isinstance(output_manifest, dict):
        raise ValueError("output_manifest must be a dictionary")
    try:
        output_json = json.dumps(output_manifest, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest must be JSON serializable") from exc
    with _connect() as conn:
        row = conn.execute(
            """SELECT status FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Engineering job not found: {job_id}")
        if row["status"] != "running":
            raise ValueError(f"Job cannot complete from status {row['status']!r}")
        conn.execute(
            """UPDATE engineering_jobs
               SET status='succeeded', output_manifest_json=?, finished_at=?
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (output_json, _now(), job_id, owner_id, workspace_id),
        )
        _event(conn, job_id, "job_completed", actor_id, {"output_keys": sorted(output_manifest)})
    return get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def fail_job(
    job_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    error: str,
) -> dict[str, Any]:
    _require_actor(actor_id)
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error is required")
    with _connect() as conn:
        row = conn.execute(
            """SELECT status FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Engineering job not found: {job_id}")
        if row["status"] not in {"running", "queued", "pending_approval"}:
            raise ValueError(f"Job cannot fail from status {row['status']!r}")
        conn.execute(
            """UPDATE engineering_jobs SET status='failed', error=?, finished_at=?
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (error[:4000], _now(), job_id, owner_id, workspace_id),
        )
        _event(conn, job_id, "job_failed", actor_id, {"error": error[:1000]})
    return get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def get_job(
    job_id: str, *, owner_id: str, workspace_id: str
) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
    return _job_dict(row)


def list_jobs(
    *, owner_id: str, workspace_id: str, status: Optional[str] = None
) -> list[dict[str, Any]]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                """SELECT * FROM engineering_jobs
                   WHERE owner_id=? AND workspace_id=? AND status=?
                   ORDER BY created_at DESC""",
                (owner_id, workspace_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM engineering_jobs
                   WHERE owner_id=? AND workspace_id=?
                   ORDER BY created_at DESC""",
                (owner_id, workspace_id),
            ).fetchall()
    return [_job_dict(row) for row in rows]  # type: ignore[list-item]


def list_job_events(
    job_id: str, *, owner_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    with _connect() as conn:
        exists = conn.execute(
            """SELECT 1 FROM engineering_jobs
               WHERE job_id=? AND owner_id=? AND workspace_id=?""",
            (job_id, owner_id, workspace_id),
        ).fetchone()
        if exists is None:
            raise KeyError(f"Engineering job not found: {job_id}")
        rows = conn.execute(
            "SELECT * FROM engineering_job_events WHERE job_id=? ORDER BY event_id",
            (job_id,),
        ).fetchall()
    events = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event.pop("details_json"))
        events.append(event)
    return events
