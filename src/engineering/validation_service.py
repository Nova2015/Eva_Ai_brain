"""Data-ID based bridge between Data Governance and engineering file validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data_governance import authorize_data_action, get_record, sha256_file
from .domain_validation import validate_engineering_file


def validate_governed_data(
    data_id: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Validate a governed local file without changing or executing the source."""
    for field, value in (
        ("data_id", data_id), ("owner_id", owner_id),
        ("workspace_id", workspace_id), ("actor_id", actor_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if actor_id != owner_id:
        raise PermissionError("Only the authenticated data owner may validate this record")
    if not authorize_data_action(
        data_id, actor_id=actor_id, workspace_id=workspace_id, action="read"
    ):
        raise PermissionError("Read permission is required for the governed source")

    record = get_record(data_id, owner_id=owner_id, workspace_id=workspace_id)
    if record is None:
        raise KeyError(f"Governed Data ID not found: {data_id}")
    path = Path(record["file_path"]).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Governed source path is not a regular file")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError("Source SHA-256 does not match the governed Data ID")

    validation = validate_engineering_file(path)
    return {
        "data_id": data_id,
        "file_name": record["file_name"],
        "sha256": digest,
        "classification": record["classification"],
        "validation": validation,
        "source_modified": False,
        "solver_execution_performed": False,
        "engineering_approval": "not_granted",
    }
