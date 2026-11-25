"""Local-first validation and quarantine pipeline for governed imports.

The validator performs safe, format-level checks only. CAD geometry, mesh quality,
FEA input, G-code semantics, and malware scanning require dedicated validators and
are reported as pending rather than falsely marked as completed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .identity_store import _connect, register_import, sha256_file


SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".step", ".stp", ".iges", ".igs", ".fcstd", ".brep",
    ".stl", ".obj", ".ply", ".3mf", ".glb", ".gltf", ".fbx", ".blend",
    ".msh", ".unv", ".inp", ".bdf", ".nas", ".foam", ".cas",
    ".gcode", ".nc", ".tap", ".ngc",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
})


def _max_import_bytes(value: Optional[int] = None) -> int:
    if value is not None:
        limit = int(value)
    else:
        try:
            limit = int(os.getenv("EVA_GOVERNANCE_MAX_IMPORT_BYTES", str(2 * 1024**3)))
        except (TypeError, ValueError):
            limit = 2 * 1024**3
    if limit < 1:
        raise ValueError("max_bytes must be a positive integer")
    return limit


def _format_check(path: Path, head: bytes) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".pdf" and not head.startswith(b"%PDF-"):
        return "PDF signature is invalid"
    if ext in {".docx", ".xlsx", ".pptx", ".fcstd", ".3mf"}:
        if not head.startswith(b"PK"):
            return f"{ext} archive signature is invalid"
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    return f"{ext} archive contains a corrupt member"
        except (OSError, zipfile.BadZipFile):
            return f"{ext} archive is corrupt"
    if ext in {".step", ".stp"}:
        if b"ISO-10303-21" not in head[:4096]:
            return "STEP exchange-file header is missing"
    if ext == ".blend" and not head.startswith(b"BLENDER"):
        return "Blender file signature is invalid"
    if ext in {".png"} and not head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG signature is invalid"
    if ext in {".jpg", ".jpeg"} and not head.startswith(b"\xff\xd8\xff"):
        return "JPEG signature is invalid"
    if ext == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "JSON content is invalid"
    if ext in {".txt", ".md", ".csv", ".jsonl", ".xml", ".yaml", ".yml", ".gcode", ".nc", ".tap", ".ngc"}:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{ext} content is not valid UTF-8 text"
    return None


def validate_import_candidate(
    file_path: str | os.PathLike[str],
    *,
    max_bytes: Optional[int] = None,
    malware_scanner: Optional[Callable[[Path], Any]] = None,
) -> dict[str, Any]:
    """Validate a local file's presence, size, extension, basic signature and hash."""
    issues: list[str] = []
    warnings: list[str] = []
    path = Path(file_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return {
            "ok": False, "status": "rejected", "issues": ["Source file does not exist"],
            "warnings": [], "file_path": str(path),
        }
    if not resolved.is_file():
        issues.append("Source path is not a regular file")
        return {
            "ok": False, "status": "rejected", "issues": issues,
            "warnings": warnings, "file_path": str(resolved),
        }

    size = resolved.stat().st_size
    limit = _max_import_bytes(max_bytes)
    if size <= 0:
        issues.append("Empty files cannot be imported")
    if size > limit:
        issues.append(f"File exceeds the configured import limit ({limit} bytes)")
    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        issues.append(f"Unsupported file extension: {ext or '[none]'}")

    head = b""
    if not issues:
        with resolved.open("rb") as handle:
            head = handle.read(4096)
        format_issue = _format_check(resolved, head)
        if format_issue:
            issues.append(format_issue)

    malware_status = "not_configured"
    if malware_scanner is None:
        warnings.append("Malware scan is not configured; antivirus validation remains pending")
    elif not issues:
        try:
            scan_result = malware_scanner(resolved)
            clean = scan_result is True or (
                isinstance(scan_result, dict) and scan_result.get("clean") is True
            )
            malware_status = "clean" if clean else "infected"
            if not clean:
                issues.append("Malware scanner did not confirm the file is clean")
        except Exception as exc:
            malware_status = "scan_error"
            issues.append(f"Malware scan failed closed: {type(exc).__name__}")

    digest = sha256_file(resolved) if resolved.is_file() else None
    return {
        "ok": not issues,
        "status": "validated" if not issues else "rejected",
        "file_path": str(resolved),
        "file_name": resolved.name,
        "extension": ext,
        "size_bytes": size,
        "sha256": digest,
        "issues": issues,
        "warnings": warnings,
        "malware_scan": malware_status,
        "deep_validation": "pending_domain_validator",
    }


def quarantine_candidate(
    file_path: str | os.PathLike[str],
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    reason: str,
    validation_report: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy a rejected candidate to local quarantine; never alter the original."""
    for field, value in (("owner_id", owner_id), ("workspace_id", workspace_id),
                         ("actor_id", actor_id), ("reason", reason)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    source = Path(file_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Quarantine source must be a regular file")

    root_value = os.getenv("EVA_DATA_QUARANTINE_DIR")
    root = Path(root_value).expanduser() if root_value else (
        Path(__file__).resolve().parents[2] / "data" / "quarantine"
    )
    root.mkdir(parents=True, exist_ok=True)
    quarantine_id = "EVA-QUAR-" + uuid.uuid4().hex.upper()
    destination = root / f"{quarantine_id}{source.suffix.lower()}"
    shutil.copy2(source, destination)

    source_hash = sha256_file(source)
    quarantine_hash = sha256_file(destination)
    if source_hash != quarantine_hash:
        destination.unlink(missing_ok=True)
        raise OSError("Quarantine copy failed SHA-256 integrity verification")

    manifest = {
        "quarantine_id": quarantine_id,
        "original_path": str(source),
        "quarantine_path": str(destination.resolve()),
        "owner_id": owner_id,
        "workspace_id": workspace_id,
        "actor_id": actor_id,
        "reason": reason.strip(),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
        "status": "quarantined",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_report": validation_report or {},
        "original_file_modified": False,
        "original_file_deleted": False,
    }
    manifest_path = root / f"{quarantine_id}.json"
    temporary_path = root / f".{quarantine_id}.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary_path, manifest_path)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO quarantine_items
               (quarantine_id, original_path, quarantine_path, owner_id, workspace_id,
                actor_id, reason, sha256, size_bytes, status, validation_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'quarantined', ?, ?)""",
            (
                quarantine_id, str(source), str(destination.resolve()), owner_id,
                workspace_id, actor_id, reason.strip(), source_hash,
                source.stat().st_size,
                json.dumps(validation_report or {}, sort_keys=True),
                manifest["created_at"],
            ),
        )
        conn.execute(
            """INSERT INTO quarantine_audit
               (quarantine_id, action, actor_id, details_json, created_at)
               VALUES (?, 'file_quarantined', ?, ?, ?)""",
            (
                quarantine_id, actor_id,
                json.dumps({"reason": reason.strip(), "sha256": source_hash}, sort_keys=True),
                manifest["created_at"],
            ),
        )
    return manifest


def list_quarantined(
    *, owner_id: str, workspace_id: str
) -> list[dict[str, Any]]:
    """List quarantined candidates only within the caller's owner/workspace scope."""
    for field, value in (("owner_id", owner_id), ("workspace_id", workspace_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM quarantine_items
               WHERE owner_id=? AND workspace_id=? AND status='quarantined'
               ORDER BY created_at DESC""",
            (owner_id, workspace_id),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["validation_report"] = json.loads(item.pop("validation_json"))
        results.append(item)
    return results


def validate_and_register_import(
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
    max_bytes: Optional[int] = None,
    malware_scanner: Optional[Callable[[Path], Any]] = None,
) -> dict[str, Any]:
    """Gate an import: valid candidates register; rejected candidates are quarantined."""
    report = validate_import_candidate(
        file_path, max_bytes=max_bytes, malware_scanner=malware_scanner
    )
    if not report["ok"]:
        quarantined = quarantine_candidate(
            file_path,
            owner_id=owner_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
            reason="; ".join(report["issues"]),
            validation_report=report,
        )
        return {
            "ok": False,
            "status": "quarantined",
            "validation": report,
            "quarantine": quarantined,
        }

    record = register_import(
        report["file_path"],
        source=source,
        owner_id=owner_id,
        workspace_id=workspace_id,
        actor_id=actor_id,
        classification=classification,
        permissions=permissions,
        project_id=project_id,
        component_id=component_id,
    )
    return {
        "ok": True,
        "status": "registered",
        "validation": report,
        "record": record,
    }
