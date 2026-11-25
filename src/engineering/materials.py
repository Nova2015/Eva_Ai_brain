"""Governed local engineering material property registry.

No material values are bundled as authoritative defaults. Every material record requires
a source reference and remains pending review until approved by a different actor.
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import tool_registry


_ALLOWED_PROPERTIES = frozenset({
    "density", "elastic_modulus", "shear_modulus", "poisson_ratio",
    "tensile_strength", "yield_strength", "compressive_strength",
    "elongation", "hardness", "impact_strength", "fatigue_strength",
    "thermal_conductivity", "specific_heat", "thermal_expansion",
    "glass_transition_temperature", "melting_temperature",
    "melt_flow_index", "moisture_absorption", "friction_coefficient",
})


def _connect() -> sqlite3.Connection:
    path = tool_registry._db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS engineering_materials (
            material_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            family TEXT NOT NULL,
            grade TEXT NOT NULL,
            manufacturer TEXT NOT NULL DEFAULT '',
            source_reference TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_review',
            created_by TEXT NOT NULL,
            approved_by TEXT,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            approved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_engineering_material_scope
            ON engineering_materials(owner_id, workspace_id, status);

        CREATE TABLE IF NOT EXISTS material_audit (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS trg_material_audit_no_update
        BEFORE UPDATE ON material_audit
        BEGIN
            SELECT RAISE(ABORT, 'material_audit is append-only');
        END;
        CREATE TRIGGER IF NOT EXISTS trg_material_audit_no_delete
        BEFORE DELETE ON material_audit
        BEGIN
            SELECT RAISE(ABORT, 'material_audit is append-only');
        END;
        """
    )
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_properties(properties: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(properties, dict) or not properties:
        raise ValueError("At least one material property is required")
    validated = {}
    for name, raw in properties.items():
        if name not in _ALLOWED_PROPERTIES:
            raise ValueError(f"Unsupported material property: {name}")
        if not isinstance(raw, dict) or "value" not in raw or "unit" not in raw:
            raise ValueError(f"{name} must contain value and unit")
        if isinstance(raw["value"], bool):
            raise ValueError(f"{name}.value must be numeric")
        try:
            value = float(raw["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.value must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}.value must be finite")
        unit = raw["unit"]
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"{name}.unit is required")
        if name == "poisson_ratio" and not -1.0 < value < 0.5:
            raise ValueError("poisson_ratio must be between -1 and 0.5 for this isotropic registry")
        if name in {
            "density", "elastic_modulus", "shear_modulus", "tensile_strength",
            "yield_strength", "compressive_strength", "fatigue_strength",
            "thermal_conductivity", "specific_heat", "melt_flow_index",
        } and value <= 0:
            raise ValueError(f"{name} must be positive")
        validated[name] = {"value": value, "unit": unit.strip()}
    return validated


def _material_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    item = dict(row)
    item["properties"] = json.loads(item.pop("properties_json"))
    return item


def register_material(
    *,
    name: str,
    family: str,
    grade: str,
    source_reference: str,
    properties: dict[str, Any],
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    manufacturer: str = "",
) -> dict[str, Any]:
    for field, value in (
        ("name", name), ("family", family), ("grade", grade),
        ("source_reference", source_reference), ("owner_id", owner_id),
        ("workspace_id", workspace_id), ("actor_id", actor_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if actor_id != owner_id:
        raise PermissionError("Only the authenticated owner may register material data in this scope")
    validated = _validate_properties(properties)
    material_id = "EVA-MAT-" + uuid.uuid4().hex.upper()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO engineering_materials
               (material_id, name, family, grade, manufacturer, source_reference,
                properties_json, owner_id, workspace_id, status, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_review', ?, ?)""",
            (
                material_id, name.strip(), family.strip(), grade.strip(),
                manufacturer.strip(), source_reference.strip(),
                json.dumps(validated, sort_keys=True), owner_id, workspace_id,
                actor_id, now,
            ),
        )
        conn.execute(
            """INSERT INTO material_audit
               (material_id, action, actor_id, details_json, created_at)
               VALUES (?, 'material_registered', ?, ?, ?)""",
            (material_id, actor_id, json.dumps({"status": "pending_review"}), now),
        )
    return get_material(material_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def get_material(
    material_id: str, *, owner_id: str, workspace_id: str
) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_materials
               WHERE material_id=? AND owner_id=? AND workspace_id=?""",
            (material_id, owner_id, workspace_id),
        ).fetchone()
    return _material_dict(row)


def list_materials(
    *, owner_id: str, workspace_id: str, status: Optional[str] = None
) -> list[dict[str, Any]]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                """SELECT * FROM engineering_materials
                   WHERE owner_id=? AND workspace_id=? AND status=?
                   ORDER BY created_at DESC""",
                (owner_id, workspace_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM engineering_materials
                   WHERE owner_id=? AND workspace_id=? ORDER BY created_at DESC""",
                (owner_id, workspace_id),
            ).fetchall()
    return [_material_dict(row) for row in rows]  # type: ignore[list-item]


def approve_material(
    material_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    approved: bool,
) -> dict[str, Any]:
    if approved is not True:
        raise ValueError("Explicit material approval is required")
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_materials
               WHERE material_id=? AND owner_id=? AND workspace_id=?""",
            (material_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Material not found: {material_id}")
        if actor_id == row["created_by"]:
            raise PermissionError("Material approval must be performed by a different reviewer")
        if row["status"] != "pending_review":
            raise ValueError(f"Material cannot be approved from status {row['status']!r}")
        now = _now()
        conn.execute(
            """UPDATE engineering_materials
               SET status='approved', approved_by=?, approved_at=?
               WHERE material_id=? AND status='pending_review'""",
            (actor_id, now, material_id),
        )
        conn.execute(
            """INSERT INTO material_audit
               (material_id, action, actor_id, details_json, created_at)
               VALUES (?, 'material_approved', ?, ?, ?)""",
            (material_id, actor_id, json.dumps({"status": "approved"}), now),
        )
    return get_material(material_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def reject_material(
    material_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    reason: str,
) -> dict[str, Any]:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A rejection reason is required")
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM engineering_materials
               WHERE material_id=? AND owner_id=? AND workspace_id=?""",
            (material_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Material not found: {material_id}")
        if actor_id == row["created_by"]:
            raise PermissionError("Material review must be performed by a different reviewer")
        if row["status"] != "pending_review":
            raise ValueError(f"Material cannot be rejected from status {row['status']!r}")
        now = _now()
        conn.execute(
            """UPDATE engineering_materials
               SET status='rejected', rejection_reason=?
               WHERE material_id=? AND status='pending_review'""",
            (reason.strip(), material_id),
        )
        conn.execute(
            """INSERT INTO material_audit
               (material_id, action, actor_id, details_json, created_at)
               VALUES (?, 'material_rejected', ?, ?, ?)""",
            (material_id, actor_id, json.dumps({"reason": reason.strip()}), now),
        )
    return get_material(material_id, owner_id=owner_id, workspace_id=workspace_id)  # type: ignore[return-value]


def resolve_material_for_analysis(
    material_id: str, *, owner_id: str, workspace_id: str
) -> dict[str, Any]:
    material = get_material(material_id, owner_id=owner_id, workspace_id=workspace_id)
    if material is None:
        raise KeyError(f"Material not found: {material_id}")
    if material["status"] != "approved":
        raise PermissionError("Only independently reviewed and approved materials may be used for analysis")
    return material


def list_material_audit(
    material_id: str, *, owner_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    with _connect() as conn:
        exists = conn.execute(
            """SELECT 1 FROM engineering_materials
               WHERE material_id=? AND owner_id=? AND workspace_id=?""",
            (material_id, owner_id, workspace_id),
        ).fetchone()
        if exists is None:
            raise KeyError(f"Material not found: {material_id}")
        rows = conn.execute(
            "SELECT * FROM material_audit WHERE material_id=? ORDER BY event_id",
            (material_id,),
        ).fetchall()
    events = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event.pop("details_json"))
        events.append(event)
    return events
