"""Local-first, Governance-linked Graph RAG storage primitives for EVA.

Graph facts are proposed by default. Retrieval helpers expose only explicitly
approved evidence, and every node/edge evidence record is linked to a governed
Data ID, owner, workspace, source version and classification.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _ROOT / "data" / "eva_graph_rag.db"
EVIDENCE_STATUSES = frozenset({"proposed", "approved", "rejected"})


def _db_path() -> Path:
    return Path(os.getenv("EVA_GRAPH_RAG_DB", str(_DEFAULT_DB))).expanduser()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        _initialize(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    yield from _connect()


def _initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_entities (
            entity_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(owner_id, workspace_id, entity_type, normalized_name)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_entities_scope
            ON graph_entities(owner_id, workspace_id, entity_type);

        CREATE TABLE IF NOT EXISTS graph_entity_evidence (
            evidence_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            data_id TEXT NOT NULL,
            source_version INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            status TEXT NOT NULL DEFAULT 'proposed'
                CHECK(status IN ('proposed', 'approved', 'rejected')),
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewed_by TEXT,
            UNIQUE(entity_id, data_id, source_version, chunk_id, evidence_sha256)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_entity_evidence_scope
            ON graph_entity_evidence(owner_id, workspace_id, data_id, status);

        CREATE TABLE IF NOT EXISTS graph_relationships (
            relationship_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            source_entity_id TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
            target_entity_id TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
            predicate TEXT NOT NULL,
            normalized_predicate TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(owner_id, workspace_id, source_entity_id, target_entity_id, normalized_predicate)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_relationships_scope
            ON graph_relationships(owner_id, workspace_id, source_entity_id, target_entity_id);

        CREATE TABLE IF NOT EXISTS graph_relationship_evidence (
            evidence_id TEXT PRIMARY KEY,
            relationship_id TEXT NOT NULL REFERENCES graph_relationships(relationship_id) ON DELETE CASCADE,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            data_id TEXT NOT NULL,
            source_version INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            status TEXT NOT NULL DEFAULT 'proposed'
                CHECK(status IN ('proposed', 'approved', 'rejected')),
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewed_by TEXT,
            UNIQUE(relationship_id, data_id, source_version, chunk_id, evidence_sha256)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_relationship_evidence_scope
            ON graph_relationship_evidence(owner_id, workspace_id, data_id, status);

        CREATE TABLE IF NOT EXISTS graph_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            action TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def _validate_scope(data_id: str, owner_id: str, workspace_id: str, actor_id: str):
    if not all(isinstance(v, str) and v.strip() for v in
               (data_id, owner_id, workspace_id, actor_id)):
        raise ValueError("data_id, owner_id, workspace_id and actor_id are required")
    try:
        from ..data_governance import get_record
    except ImportError:
        from data_governance import get_record
    record = get_record(data_id, owner_id=owner_id, workspace_id=workspace_id)
    if not record:
        raise PermissionError("Source Data ID is not available in this owner/workspace")
    return record


def _validate_evidence(evidence_text: str, confidence: float) -> tuple[str, str]:
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError("evidence_text is required")
    if len(evidence_text) > 2000:
        raise ValueError("evidence_text must be 2000 characters or fewer")
    score = float(confidence)
    if not 0.0 <= score <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    snippet = evidence_text.strip()
    return snippet, hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def _audit(conn: sqlite3.Connection, owner_id: str, workspace_id: str,
           actor_id: str, action: str, object_type: str, object_id: str,
           details: Optional[dict[str, Any]] = None) -> None:
    conn.execute(
        """INSERT INTO graph_audit
           (owner_id, workspace_id, actor_id, action, object_type, object_id, details_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (owner_id, workspace_id, actor_id, action, object_type, object_id,
         json.dumps(details or {}, sort_keys=True), _now()),
    )


def upsert_entity(
    name: str,
    entity_type: str,
    *,
    data_id: str,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    chunk_id: str,
    evidence_text: str,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Create/reuse an entity and attach proposed, source-versioned evidence."""
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 300:
        raise ValueError("name must contain 1-300 characters")
    if not isinstance(entity_type, str) or not entity_type.strip() or len(entity_type.strip()) > 80:
        raise ValueError("entity_type must contain 1-80 characters")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("chunk_id is required")
    record = _validate_scope(data_id, owner_id, workspace_id, actor_id)
    snippet, digest = _validate_evidence(evidence_text, confidence)
    now = _now()
    normalized_name = _normalize(name)
    normalized_type = _normalize(entity_type)
    entity_id = "EVA-ENT-" + uuid.uuid4().hex.upper()
    evidence_id = "EVA-GE-" + uuid.uuid4().hex.upper()

    with _connection() as conn:
        conn.execute(
            """INSERT INTO graph_entities
               (entity_id, owner_id, workspace_id, entity_type, canonical_name,
                normalized_name, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(owner_id, workspace_id, entity_type, normalized_name)
               DO UPDATE SET canonical_name=excluded.canonical_name, updated_at=excluded.updated_at""",
            (entity_id, owner_id, workspace_id, normalized_type, name.strip(),
             normalized_name, now, now),
        )
        row = conn.execute(
            """SELECT entity_id FROM graph_entities
               WHERE owner_id=? AND workspace_id=? AND entity_type=? AND normalized_name=?""",
            (owner_id, workspace_id, normalized_type, normalized_name),
        ).fetchone()
        entity_id = row["entity_id"]
        conn.execute(
            """INSERT OR IGNORE INTO graph_entity_evidence
               (evidence_id, entity_id, owner_id, workspace_id, data_id, source_version,
                chunk_id, evidence_text, evidence_sha256, classification, confidence, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?)""",
            (evidence_id, entity_id, owner_id, workspace_id, data_id,
             int(record.get("version") or 1), chunk_id.strip(), snippet, digest,
             record.get("classification") or "internal", float(confidence), now),
        )
        evidence = conn.execute(
            """SELECT evidence_id, status FROM graph_entity_evidence
               WHERE entity_id=? AND data_id=? AND chunk_id=? AND evidence_sha256=?""",
            (entity_id, data_id, chunk_id.strip(), digest),
        ).fetchone()
        _audit(conn, owner_id, workspace_id, actor_id, "entity_evidence_proposed",
               "entity", entity_id, {"data_id": data_id, "evidence_id": evidence["evidence_id"]})
    return {"entity_id": entity_id, "evidence_id": evidence["evidence_id"],
            "entity_type": normalized_type, "name": name.strip(), "status": evidence["status"]}


def upsert_relationship(
    source_entity_id: str,
    target_entity_id: str,
    predicate: str,
    *,
    data_id: str,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    chunk_id: str,
    evidence_text: str,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Create/reuse a directed relationship; its evidence starts as proposed."""
    if not all(isinstance(v, str) and v.strip() for v in
               (source_entity_id, target_entity_id, predicate, chunk_id)):
        raise ValueError("source_entity_id, target_entity_id, predicate and chunk_id are required")
    if len(predicate.strip()) > 120:
        raise ValueError("predicate must be 120 characters or fewer")
    record = _validate_scope(data_id, owner_id, workspace_id, actor_id)
    snippet, digest = _validate_evidence(evidence_text, confidence)
    now = _now()
    relationship_id = "EVA-REL-" + uuid.uuid4().hex.upper()
    evidence_id = "EVA-GRE-" + uuid.uuid4().hex.upper()
    normalized_predicate = _normalize(predicate)

    with _connection() as conn:
        endpoints = conn.execute(
            """SELECT entity_id FROM graph_entities
               WHERE owner_id=? AND workspace_id=? AND entity_id IN (?, ?)""",
            (owner_id, workspace_id, source_entity_id, target_entity_id),
        ).fetchall()
        if len({row["entity_id"] for row in endpoints}) != 2:
            raise PermissionError("Both relationship endpoints must belong to the same owner/workspace")
        conn.execute(
            """INSERT OR IGNORE INTO graph_relationships
               (relationship_id, owner_id, workspace_id, source_entity_id, target_entity_id,
                predicate, normalized_predicate, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (relationship_id, owner_id, workspace_id, source_entity_id, target_entity_id,
             predicate.strip(), normalized_predicate, now),
        )
        row = conn.execute(
            """SELECT relationship_id FROM graph_relationships
               WHERE owner_id=? AND workspace_id=? AND source_entity_id=?
                 AND target_entity_id=? AND normalized_predicate=?""",
            (owner_id, workspace_id, source_entity_id, target_entity_id, normalized_predicate),
        ).fetchone()
        relationship_id = row["relationship_id"]
        conn.execute(
            """INSERT OR IGNORE INTO graph_relationship_evidence
               (evidence_id, relationship_id, owner_id, workspace_id, data_id, source_version,
                chunk_id, evidence_text, evidence_sha256, classification, confidence, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?)""",
            (evidence_id, relationship_id, owner_id, workspace_id, data_id,
             int(record.get("version") or 1), chunk_id.strip(), snippet, digest,
             record.get("classification") or "internal", float(confidence), now),
        )
        evidence = conn.execute(
            """SELECT evidence_id, status FROM graph_relationship_evidence
               WHERE relationship_id=? AND data_id=? AND chunk_id=? AND evidence_sha256=?""",
            (relationship_id, data_id, chunk_id.strip(), digest),
        ).fetchone()
        _audit(conn, owner_id, workspace_id, actor_id, "relationship_evidence_proposed",
               "relationship", relationship_id,
               {"data_id": data_id, "evidence_id": evidence["evidence_id"]})
    return {"relationship_id": relationship_id, "evidence_id": evidence["evidence_id"],
            "predicate": predicate.strip(), "status": evidence["status"]}


def review_evidence(
    evidence_id: str,
    status: str,
    *,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Explicitly approve/reject one graph evidence record within its owner scope."""
    if status not in {"approved", "rejected"}:
        raise ValueError("status must be approved or rejected")
    if not all(isinstance(v, str) and v.strip() for v in
               (evidence_id, owner_id, workspace_id, actor_id)):
        raise ValueError("evidence_id, owner_id, workspace_id and actor_id are required")
    now = _now()
    with _connection() as conn:
        row = conn.execute(
            """SELECT evidence_id, data_id, source_version FROM graph_entity_evidence
               WHERE evidence_id=? AND owner_id=? AND workspace_id=?""",
            (evidence_id, owner_id, workspace_id),
        ).fetchone()
        table = "graph_entity_evidence"
        object_type = "entity_evidence"
        if row is None:
            row = conn.execute(
                """SELECT evidence_id, data_id, source_version FROM graph_relationship_evidence
                   WHERE evidence_id=? AND owner_id=? AND workspace_id=?""",
                (evidence_id, owner_id, workspace_id),
            ).fetchone()
            table = "graph_relationship_evidence"
            object_type = "relationship_evidence"
        if row is None:
            raise KeyError("Graph evidence not found")
        if status == "approved":
            source_record = _validate_scope(
                row["data_id"], owner_id, workspace_id, actor_id
            )
            if int(source_record.get("version") or 1) != int(row["source_version"]):
                raise ValueError("Source Data ID has a newer revision; re-extract graph evidence before approval")
        conn.execute(
            f"UPDATE {table} SET status=?, reviewed_at=?, reviewed_by=? WHERE evidence_id=? AND owner_id=? AND workspace_id=?",
            (status, now, actor_id, evidence_id, owner_id, workspace_id),
        )
        _audit(conn, owner_id, workspace_id, actor_id, f"graph_evidence_{status}",
               object_type, evidence_id, {})
    return {"evidence_id": evidence_id, "status": status, "reviewed_by": actor_id}


def _evidence_matches_current_revision(data_id: str, owner_id: str, workspace_id: str, source_version: int) -> bool:
    """Hide approved graph evidence when its governed source has advanced."""
    try:
        from ..data_governance import get_record
    except ImportError:
        from data_governance import get_record
    record = get_record(data_id)
    return bool(
        record
        and record.get("owner_id") == owner_id
        and record.get("workspace_id") == workspace_id
        and int(record.get("version") or 1) == int(source_version)
    )


def search_entities(
    query: str,
    *,
    owner_id: str,
    workspace_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find only entities backed by approved evidence in the requested scope."""
    if not query or not query.strip():
        return []
    if not owner_id or not workspace_id:
        raise ValueError("owner_id and workspace_id are required")
    safe_limit = min(max(int(limit), 1), 100)
    term = "%" + _normalize(query) + "%"
    with _connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT n.entity_id, n.entity_type, n.canonical_name,
                      e.data_id, e.source_version
               FROM graph_entities n
               JOIN graph_entity_evidence e ON e.entity_id=n.entity_id
               WHERE n.owner_id=? AND n.workspace_id=? AND e.owner_id=? AND e.workspace_id=?
                 AND e.status='approved' AND n.normalized_name LIKE ?
               ORDER BY n.canonical_name LIMIT ?""",
            (owner_id, workspace_id, owner_id, workspace_id, term, safe_limit * 5),
        ).fetchall()
    current = {}
    results = {}
    for row in rows:
        data_id = row["data_id"]
        if data_id not in current:
            current[data_id] = _evidence_matches_current_revision(
                data_id, owner_id, workspace_id, row["source_version"]
            )
        if current[data_id]:
            results[row["entity_id"]] = {
                "entity_id": row["entity_id"],
                "entity_type": row["entity_type"],
                "canonical_name": row["canonical_name"],
            }
            if len(results) >= safe_limit:
                break
    return list(results.values())


def traverse_verified_relationships(
    seed_entity_ids: list[str],
    *,
    owner_id: str,
    workspace_id: str,
    max_hops: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Traverse approved edges only; every returned edge carries evidence provenance."""
    if not owner_id or not workspace_id:
        raise ValueError("owner_id and workspace_id are required")
    if not isinstance(seed_entity_ids, list) or not seed_entity_ids:
        return []
    hops = min(max(int(max_hops), 1), 5)
    safe_limit = min(max(int(limit), 1), 200)
    frontier = set(str(x) for x in seed_entity_ids if str(x))
    visited = set(frontier)
    found: dict[str, dict[str, Any]] = {}
    with _connection() as conn:
        for depth in range(1, hops + 1):
            if not frontier or len(found) >= safe_limit:
                break
            ids = list(frontier)
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                           r.predicate, src.canonical_name AS source_name,
                           dst.canonical_name AS target_name, e.evidence_id, e.data_id,
                           e.source_version, e.chunk_id, e.evidence_text, e.confidence,
                           e.classification
                    FROM graph_relationships r
                    JOIN graph_relationship_evidence e ON e.relationship_id=r.relationship_id
                    JOIN graph_entities src ON src.entity_id=r.source_entity_id
                    JOIN graph_entities dst ON dst.entity_id=r.target_entity_id
                    WHERE r.owner_id=? AND r.workspace_id=? AND e.owner_id=? AND e.workspace_id=?
                      AND e.status='approved'
                      AND (r.source_entity_id IN ({placeholders}) OR r.target_entity_id IN ({placeholders}))
                    LIMIT ?""",
                (owner_id, workspace_id, owner_id, workspace_id, *ids, *ids, safe_limit),
            ).fetchall()
            next_frontier = set()
            for row in rows:
                item = dict(row)
                item["depth"] = depth
                found[item["relationship_id"] + ":" + item["evidence_id"]] = item
                for endpoint in (item["source_entity_id"], item["target_entity_id"]):
                    if endpoint not in visited:
                        visited.add(endpoint)
                        next_frontier.add(endpoint)
                if len(found) >= safe_limit:
                    break
            frontier = next_frontier
    current = {}
    verified = []
    for item in found.values():
        data_id = item["data_id"]
        if data_id not in current:
            current[data_id] = _evidence_matches_current_revision(
                data_id, owner_id, workspace_id, item["source_version"]
            )
        if current[data_id]:
            verified.append(item)
    return verified


def list_graph_audit(
    *,
    owner_id: str,
    workspace_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not owner_id or not workspace_id:
        raise ValueError("owner_id and workspace_id are required")
    safe_limit = min(max(int(limit), 1), 500)
    with _connection() as conn:
        rows = conn.execute(
            """SELECT actor_id, action, object_type, object_id, details_json, created_at
               FROM graph_audit WHERE owner_id=? AND workspace_id=?
               ORDER BY audit_id DESC LIMIT ?""",
            (owner_id, workspace_id, safe_limit),
        ).fetchall()
    return [{
        "actor_id": row["actor_id"], "action": row["action"],
        "object_type": row["object_type"], "object_id": row["object_id"],
        "details": json.loads(row["details_json"]), "created_at": row["created_at"],
    } for row in rows]
