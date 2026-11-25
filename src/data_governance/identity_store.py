"""Persistent, local-first identity and governance metadata for EVA data items.

A Data ID identifies an imported item, not its filename or content hash.
Consequently, two separately imported copies may share a SHA-256 hash while
retaining distinct Data IDs. Renames and moves update the location without
changing identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

CLASSIFICATIONS = frozenset({"public", "internal", "confidential", "restricted"})
TRAINING_APPROVALS = frozenset({"pending", "approved", "rejected"})

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _ROOT / "data" / "eva_data_governance.db"


def _db_path() -> Path:
    return Path(os.getenv("EVA_DATA_GOVERNANCE_DB", str(_DEFAULT_DB))).expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_id() -> str:
    return f"EVA-DATA-{uuid.uuid4().hex.upper()}"


def sha256_file(file_path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        _initialize(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS data_items (
            data_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            classification TEXT NOT NULL,
            permissions_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
            project_id TEXT,
            component_id TEXT,
            training_approval TEXT NOT NULL DEFAULT 'pending',
            created_by TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_data_items_workspace
            ON data_items(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_data_items_sha256
            ON data_items(sha256);
        CREATE INDEX IF NOT EXISTS idx_data_items_project
            ON data_items(project_id, component_id);

        CREATE TABLE IF NOT EXISTS data_item_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id TEXT NOT NULL REFERENCES data_items(data_id) ON DELETE CASCADE,
            version INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            changed_by TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            UNIQUE(data_id, version)
        );

        CREATE TABLE IF NOT EXISTS reconciliation_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS data_item_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id TEXT NOT NULL REFERENCES data_items(data_id) ON DELETE CASCADE,
            action TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def _audit(
    conn: sqlite3.Connection,
    data_id: str,
    action: str,
    actor_id: str,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT INTO data_item_audit
           (data_id, action, actor_id, details_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (data_id, action, actor_id, json.dumps(details, sort_keys=True), _now()),
    )


def _row_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    result = dict(row)
    result["permissions"] = json.loads(result.pop("permissions_json"))
    return result


def _require_scope(data_id: str, owner_id: str, workspace_id: str) -> None:
    """Reject missing scope before any data-item read or mutation."""
    for field, value in (("data_id", data_id), ("owner_id", owner_id),
                         ("workspace_id", workspace_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")


_PERMISSION_ACTIONS = ("read", "write", "execute", "delete")


def _normalize_permissions(
    permissions: Optional[dict[str, Any]],
    *,
    default_owner: Optional[str] = None,
) -> dict[str, list[str]]:
    """Validate and normalize the four independent permission capabilities."""
    if permissions is None:
        permissions = {}
        if default_owner:
            permissions = {"read": [default_owner], "write": [default_owner]}
    if not isinstance(permissions, dict):
        raise ValueError("permissions must be a dictionary")
    unknown = set(permissions) - set(_PERMISSION_ACTIONS)
    if unknown:
        raise ValueError(f"Unsupported permission actions: {sorted(unknown)}")

    normalized: dict[str, list[str]] = {}
    for action in _PERMISSION_ACTIONS:
        principals = permissions.get(action, [])
        if not isinstance(principals, list):
            raise ValueError(f"permissions[{action!r}] must be a list of principal IDs")
        if any(not isinstance(item, str) or not item.strip() for item in principals):
            raise ValueError(f"permissions[{action!r}] contains an invalid principal ID")
        normalized[action] = sorted(set(principals))
    return normalized


def _authorized_record(
    conn: sqlite3.Connection,
    data_id: str,
    actor_id: str,
    workspace_id: str,
    action: str,
) -> Optional[sqlite3.Row]:
    """Resolve a record only when its owner or ACL grants the requested action."""
    if action not in _PERMISSION_ACTIONS:
        raise ValueError(f"Unsupported permission action: {action}")
    row = conn.execute(
        "SELECT * FROM data_items WHERE data_id=? AND workspace_id=?",
        (data_id, workspace_id),
    ).fetchone()
    if row is None:
        return None
    if row["owner_id"] == actor_id:
        return row
    permissions = json.loads(row["permissions_json"])
    return row if actor_id in permissions.get(action, []) else None


def authorize_data_action(
    data_id: str, *, actor_id: str, workspace_id: str, action: str
) -> bool:
    """Check one of read/write/execute/delete against the item's owner and ACL."""
    _require_scope(data_id, actor_id, workspace_id)
    with _connect() as conn:
        return _authorized_record(conn, data_id, actor_id, workspace_id, action) is not None


def register_import(
    file_path: str | os.PathLike[str],
    *,
    source: str,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    classification: str = "internal",
    permissions: Optional[dict[str, Any]] = None,
    project_id: Optional[str] = None,
    component_id: Optional[str] = None,
) -> dict[str, Any]:
    """Register an existing local file as a new, independently identified item."""
    path = Path(file_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("file_path must point to a regular file")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"Unsupported classification: {classification}")
    for field, value in (("source", source), ("owner_id", owner_id),
                         ("workspace_id", workspace_id), ("actor_id", actor_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")

    digest = sha256_file(path)
    size = path.stat().st_size
    now = _now()
    data_id = _data_id()
    permission_data = _normalize_permissions(permissions, default_owner=owner_id)

    with _connect() as conn:
        conn.execute(
            """INSERT INTO data_items
               (data_id, source, file_name, file_path, owner_id, workspace_id,
                imported_at, sha256, size_bytes, classification,
                permissions_json, version, project_id, component_id,
                training_approval, created_by, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'pending', ?, ?)""",
            (data_id, source, path.name, str(path), owner_id, workspace_id,
             now, digest, size, classification,
             json.dumps(permission_data, sort_keys=True), project_id,
             component_id, actor_id, now),
        )
        conn.execute(
            """INSERT INTO data_item_versions
               (data_id, version, sha256, size_bytes, file_path, changed_by, changed_at)
               VALUES (?, 1, ?, ?, ?, ?, ?)""",
            (data_id, digest, size, str(path), actor_id, now),
        )
        _audit(conn, data_id, "import_registered", actor_id, {
            "source": source, "classification": classification,
            "workspace_id": workspace_id, "sha256": digest,
        })
        row = conn.execute("SELECT * FROM data_items WHERE data_id = ?", (data_id,)).fetchone()
    return _row_dict(row)  # type: ignore[return-value]


def list_records(
    *,
    owner_id: str,
    workspace_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List governance records within an authenticated owner's scope."""
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("owner_id is required")
    with _connect() as conn:
        if workspace_id is None:
            rows = conn.execute(
                "SELECT * FROM data_items WHERE owner_id=? ORDER BY imported_at",
                (owner_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM data_items
                   WHERE owner_id=? AND workspace_id=? ORDER BY imported_at""",
                (owner_id, workspace_id),
            ).fetchall()
    return [_row_dict(row) for row in rows]  # type: ignore[misc]


def get_record(
    data_id: str, *, owner_id: str, workspace_id: str
) -> Optional[dict[str, Any]]:
    """Return a record only inside the authenticated owner/workspace scope."""
    _require_scope(data_id, owner_id, workspace_id)
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "read")
    return _row_dict(row)


def update_location(
    data_id: str,
    new_file_path: str | os.PathLike[str],
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Update a moved/renamed file only within the caller's owner/workspace scope."""
    _require_scope(data_id, owner_id, workspace_id)
    path = Path(new_file_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("new_file_path must point to a regular file")
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown governed Data ID: {data_id}")
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError("Moved file content does not match the governed SHA-256 revision")
        size = path.stat().st_size
        old_path = row["file_path"]
        conn.execute(
            """UPDATE data_items SET file_path=?, file_name=?, size_bytes=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (str(path), path.name, size, _now(), data_id, workspace_id),
        )
        _audit(conn, data_id, "location_updated", actor_id, {
            "old_path": old_path, "new_path": str(path), "sha256": digest,
        })
        updated = conn.execute(
            """SELECT * FROM data_items WHERE data_id=? AND workspace_id=?""",
            (data_id, workspace_id),
        ).fetchone()
    return _row_dict(updated)  # type: ignore[return-value]


def recover_location(
    data_id: str,
    new_file_path: str | os.PathLike[str],
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    allowed_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recover a missing governed file location without changing its identity.

    The candidate must remain inside the configured governed-document root and
    its bytes must match the currently approved SHA-256 revision.
    """
    if not all(isinstance(value, str) and value.strip() for value in
               (data_id, owner_id, workspace_id, actor_id)):
        raise ValueError("data_id, owner_id, workspace_id and actor_id are required")
    root = Path(allowed_root).expanduser().resolve(strict=True)
    candidate = Path(new_file_path).expanduser().resolve(strict=True)
    if not root.is_dir() or not candidate.is_file():
        raise ValueError("Recovery root and candidate must be an existing directory and file")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Recovery candidate must be inside the governed document directory") from exc

    digest = sha256_file(candidate)
    size = candidate.stat().st_size
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown governed Data ID: {data_id}")
        if digest != row["sha256"]:
            raise ValueError("Candidate content does not match the governed SHA-256 revision")
        old_path = row["file_path"]
        conn.execute(
            """UPDATE data_items
               SET file_path=?, file_name=?, size_bytes=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (str(candidate), candidate.name, size, _now(), data_id, workspace_id),
        )
        _audit(conn, data_id, "location_recovered", actor_id, {
            "old_path": old_path,
            "new_path": str(candidate),
            "sha256": digest,
        })
        updated = conn.execute(
            "SELECT * FROM data_items WHERE data_id=?", (data_id,)
        ).fetchone()
    return _row_dict(updated)  # type: ignore[return-value]


def record_revision(
    data_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Record a new content version only within the caller's scope."""
    _require_scope(data_id, owner_id, workspace_id)
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        path = Path(row["file_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Registered file is missing: {path}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest == row["sha256"]:
            return {"data_id": data_id, "version": row["version"],
                    "sha256": digest, "changed": False}
        next_version = int(row["version"]) + 1
        now = _now()
        conn.execute(
            """UPDATE data_items SET sha256=?, size_bytes=?, version=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (digest, size, next_version, now, data_id, workspace_id),
        )
        conn.execute(
            """INSERT INTO data_item_versions
               (data_id, version, sha256, size_bytes, file_path, changed_by, changed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data_id, next_version, digest, size, str(path), actor_id, now),
        )
        _audit(conn, data_id, "content_revision_recorded", actor_id, {
            "version": next_version, "sha256": digest, "size_bytes": size,
        })
        return {"data_id": data_id, "version": next_version,
                "sha256": digest, "changed": True}


def approve_content_revision(
    data_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Approve changed local bytes as the next revision for the scoped owner."""
    if not all(isinstance(value, str) and value.strip() for value in
               (data_id, owner_id, workspace_id, actor_id, expected_sha256)):
        raise ValueError("data_id, owner_id, workspace_id, actor_id and expected_sha256 are required")
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown governed Data ID: {data_id}")
        if row["sha256"] != expected_sha256:
            raise ValueError("Governance revision changed; refresh reconciliation before approval")
        path = Path(row["file_path"])
        if not path.is_file():
            raise FileNotFoundError("Governed source file is missing")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest == row["sha256"]:
            return {
                "data_id": data_id,
                "version": row["version"],
                "sha256": digest,
                "changed": False,
            }
        next_version = int(row["version"]) + 1
        now = _now()
        conn.execute(
            """UPDATE data_items SET sha256=?, size_bytes=?, version=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (digest, size, next_version, now, data_id, workspace_id),
        )
        conn.execute(
            """INSERT INTO data_item_versions
               (data_id, version, sha256, size_bytes, file_path, changed_by, changed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data_id, next_version, digest, size, str(path), actor_id, now),
        )
        _audit(conn, data_id, "content_revision_approved", actor_id, {
            "version": next_version,
            "previous_sha256": row["sha256"],
            "sha256": digest,
            "size_bytes": size,
        })
        return {
            "data_id": data_id,
            "version": next_version,
            "sha256": digest,
            "changed": True,
        }


def set_classification(
    data_id: str,
    classification: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    _require_scope(data_id, owner_id, workspace_id)
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"Unsupported classification: {classification}")
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        previous = row["classification"]
        conn.execute(
            """UPDATE data_items SET classification=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (classification, _now(), data_id, workspace_id),
        )
        _audit(conn, data_id, "classification_changed", actor_id,
               {"from": previous, "to": classification})
        updated = conn.execute(
            """SELECT * FROM data_items WHERE data_id=? AND workspace_id=?""",
            (data_id, workspace_id),
        ).fetchone()
    return _row_dict(updated)  # type: ignore[return-value]


def set_permissions(
    data_id: str,
    permissions: dict[str, Any],
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    _require_scope(data_id, owner_id, workspace_id)
    normalized_permissions = _normalize_permissions(permissions)
    with _connect() as conn:
        row = conn.execute(
            """SELECT data_id FROM data_items
               WHERE data_id=? AND owner_id=? AND workspace_id=?""",
            (data_id, owner_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        conn.execute(
            """UPDATE data_items SET permissions_json=?, updated_at=?
               WHERE data_id=? AND owner_id=? AND workspace_id=?""",
            (json.dumps(normalized_permissions, sort_keys=True), _now(), data_id, owner_id, workspace_id),
        )
        _audit(conn, data_id, "permissions_changed", actor_id, {"permissions": normalized_permissions})
        updated = conn.execute(
            """SELECT * FROM data_items WHERE data_id=? AND workspace_id=?""",
            (data_id, workspace_id),
        ).fetchone()
    return _row_dict(updated)  # type: ignore[return-value]


def set_training_approval(
    data_id: str,
    status: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Set explicit human training-data approval inside the caller's scope."""
    _require_scope(data_id, owner_id, workspace_id)
    if status not in TRAINING_APPROVALS:
        raise ValueError(f"Unsupported training approval: {status}")
    if status == "approved" and (not isinstance(actor_id, str) or not actor_id.strip()):
        raise ValueError("An approving actor_id is required")
    with _connect() as conn:
        row = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if row is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        previous = row["training_approval"]
        conn.execute(
            """UPDATE data_items SET training_approval=?, updated_at=?
               WHERE data_id=? AND workspace_id=?""",
            (status, _now(), data_id, workspace_id),
        )
        _audit(conn, data_id, "training_approval_changed", actor_id,
               {"from": previous, "to": status})
        updated = conn.execute(
            """SELECT * FROM data_items WHERE data_id=? AND workspace_id=?""",
            (data_id, workspace_id),
        ).fetchone()
    return _row_dict(updated)  # type: ignore[return-value]


def list_versions(
    data_id: str, *, owner_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    """List versions only for a Data ID in the caller's scope."""
    _require_scope(data_id, owner_id, workspace_id)
    with _connect() as conn:
        exists = _authorized_record(conn, data_id, owner_id, workspace_id, "read")
        if exists is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        rows = conn.execute(
            """SELECT version, sha256, size_bytes, file_path, changed_by, changed_at
               FROM data_item_versions WHERE data_id=? ORDER BY version""",
            (data_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def record_governance_event(
    data_id: str,
    action: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Append an audit event only when the Data ID belongs to the caller scope."""
    if not all(isinstance(value, str) and value.strip() for value in
               (data_id, action, owner_id, workspace_id, actor_id)):
        raise ValueError("data_id, action, owner_id, workspace_id and actor_id are required")
    with _connect() as conn:
        exists = _authorized_record(conn, data_id, owner_id, workspace_id, "write")
        if exists is None:
            raise KeyError(f"Unknown governed Data ID: {data_id}")
        _audit(conn, data_id, action, actor_id, details or {})


def record_reconciliation_event(
    data_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    action: str,
    actor_id: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Audit reconciliation actions, including orphan IDs without Governance rows."""
    if not all(isinstance(value, str) and value.strip() for value in
               (data_id, owner_id, workspace_id, action, actor_id)):
        raise ValueError("data_id, owner_id, workspace_id, action and actor_id are required")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO reconciliation_audit
               (data_id, owner_id, workspace_id, action, actor_id, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data_id, owner_id, workspace_id, action, actor_id,
             json.dumps(details or {}, sort_keys=True), _now()),
        )


def list_reconciliation_events(
    *,
    owner_id: str,
    workspace_id: str,
    data_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Read reconciliation audit events for the authenticated owner/workspace."""
    if not all(isinstance(value, str) and value.strip() for value in (owner_id, workspace_id)):
        raise ValueError("owner_id and workspace_id are required")
    with _connect() as conn:
        if data_id:
            rows = conn.execute(
                """SELECT data_id, action, actor_id, details_json, created_at
                   FROM reconciliation_audit
                   WHERE owner_id=? AND workspace_id=? AND data_id=?
                   ORDER BY audit_id""",
                (owner_id, workspace_id, data_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT data_id, action, actor_id, details_json, created_at
                   FROM reconciliation_audit
                   WHERE owner_id=? AND workspace_id=?
                   ORDER BY audit_id""",
                (owner_id, workspace_id),
            ).fetchall()
    return [{
        "data_id": row["data_id"],
        "action": row["action"],
        "actor_id": row["actor_id"],
        "details": json.loads(row["details_json"]),
        "created_at": row["created_at"],
    } for row in rows]


def list_audit_events(
    data_id: str, *, owner_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    """Read the append-only audit trail only inside the caller's scope."""
    _require_scope(data_id, owner_id, workspace_id)
    with _connect() as conn:
        exists = _authorized_record(conn, data_id, owner_id, workspace_id, "read")
        if exists is None:
            raise KeyError(f"Unknown Data ID: {data_id}")
        rows = conn.execute(
            """SELECT action, actor_id, details_json, created_at
               FROM data_item_audit WHERE data_id=? ORDER BY audit_id""",
            (data_id,),
        ).fetchall()
    return [
        {
            "action": row["action"],
            "actor_id": row["actor_id"],
            "details": json.loads(row["details_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
