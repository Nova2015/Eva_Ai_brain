"""SQLite-backed dataset, training-job and model metadata registry."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _ROOT / "data" / "eva_deep_learning.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeepLearningRegistry:
    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = Path(db_path or os.getenv("EVA_DL_REGISTRY_DB", str(_DEFAULT_DB))).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS dl_datasets (
                    dataset_id TEXT PRIMARY KEY, data_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    file_name TEXT NOT NULL, file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, classification TEXT NOT NULL,
                    source_version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_dl_dataset_scope
                    ON dl_datasets(owner_id, workspace_id, created_at);
                CREATE TABLE IF NOT EXISTS dl_training_jobs (
                    job_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                    data_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL, status TEXT NOT NULL,
                    stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    selected_device TEXT NOT NULL, execution_environment TEXT NOT NULL,
                    error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(dataset_id) REFERENCES dl_datasets(dataset_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dl_jobs_scope
                    ON dl_training_jobs(owner_id, workspace_id, created_at);
                CREATE TABLE IF NOT EXISTS dl_models (
                    model_id TEXT NOT NULL, version INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL, data_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    model_family TEXT NOT NULL, task_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL, metrics_json TEXT NOT NULL,
                    preprocessing_json TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(model_id, version)
                );
            """)

    def create_dataset(self, *, data_id: str, owner_id: str, workspace_id: str,
                       file_name: str, file_path: str, file_type: str,
                       size_bytes: int, sha256: str, classification: str,
                       source_version: int = 1, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        item = {
            "dataset_id": f"EVA-DSET-{uuid.uuid4().hex.upper()}",
            "data_id": data_id, "owner_id": owner_id, "workspace_id": workspace_id,
            "file_name": file_name, "file_path": file_path, "file_type": file_type,
            "size_bytes": int(size_bytes), "sha256": sha256,
            "classification": classification, "source_version": int(source_version),
            "status": "awaiting_training_approval", "created_at": _now(),
            "metadata_json": json.dumps(metadata or {}, sort_keys=True),
        }
        with self.connect() as conn:
            conn.execute("""INSERT INTO dl_datasets
                (dataset_id,data_id,owner_id,workspace_id,file_name,file_path,file_type,
                 size_bytes,sha256,classification,source_version,status,created_at,metadata_json)
                VALUES (:dataset_id,:data_id,:owner_id,:workspace_id,:file_name,:file_path,
                 :file_type,:size_bytes,:sha256,:classification,:source_version,:status,
                 :created_at,:metadata_json)""", item)
        return self._public_dataset(item)

    @staticmethod
    def _public_dataset(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result.pop("file_path", None)
        raw = result.pop("metadata_json", "{}")
        result["metadata"] = json.loads(raw) if isinstance(raw, str) else raw
        return result

    def get_dataset(self, dataset_id: str, *, owner_id: str, workspace_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM dl_datasets
                WHERE dataset_id=? AND owner_id=? AND workspace_id=?""",
                (dataset_id, owner_id, workspace_id)).fetchone()
        return dict(row) if row else None

    def list_datasets(self, *, owner_id: str, workspace_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT * FROM dl_datasets
                WHERE owner_id=? AND workspace_id=? ORDER BY created_at DESC""",
                (owner_id, workspace_id)).fetchall()
        return [self._public_dataset(dict(row)) for row in rows]

    def approve_dataset(self, dataset_id: str, *, owner_id: str, workspace_id: str,
                        device: str, environment: str = "local") -> dict[str, Any]:
        now = _now()
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM dl_datasets
                WHERE dataset_id=? AND owner_id=? AND workspace_id=?""",
                (dataset_id, owner_id, workspace_id)).fetchone()
            if row is None:
                raise KeyError("Dataset not found")
            if row["status"] not in {"awaiting_training_approval", "prepared", "reviewed", "approved"}:
                raise ValueError("Dataset is not eligible for training")
            conn.execute("UPDATE dl_datasets SET status='approved' WHERE dataset_id=?", (dataset_id,))
            existing = conn.execute("""SELECT * FROM dl_training_jobs
                WHERE dataset_id=? AND owner_id=? AND workspace_id=? AND status IN ('queued','running')
                ORDER BY created_at DESC LIMIT 1""", (dataset_id, owner_id, workspace_id)).fetchone()
            if existing:
                return dict(existing)
            job = {
                "job_id": f"EVA-DL-JOB-{uuid.uuid4().hex.upper()}",
                "dataset_id": dataset_id, "data_id": row["data_id"],
                "owner_id": owner_id, "workspace_id": workspace_id,
                "status": "queued", "stage": "queued_for_training_worker",
                "progress": 0.0, "selected_device": device,
                "execution_environment": environment, "error_code": None,
                "created_at": now, "updated_at": now, "metadata_json": "{}",
            }
            conn.execute("""INSERT INTO dl_training_jobs
                (job_id,dataset_id,data_id,owner_id,workspace_id,status,stage,progress,
                 selected_device,execution_environment,error_code,created_at,updated_at,metadata_json)
                VALUES (:job_id,:dataset_id,:data_id,:owner_id,:workspace_id,:status,:stage,:progress,
                 :selected_device,:execution_environment,:error_code,:created_at,:updated_at,:metadata_json)""", job)
        return job

    def get_job(self, job_id: str, *, owner_id: str, workspace_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM dl_training_jobs
                WHERE job_id=? AND owner_id=? AND workspace_id=?""",
                (job_id, owner_id, workspace_id)).fetchone()
        if not row:
            return None
        item = dict(row)
        item.pop("metadata_json", None)
        return item

    def update_dataset_metadata(self, dataset_id: str, *, owner_id: str,
                                workspace_id: str, updates: dict[str, Any],
                                status: str | None = None) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM dl_datasets
                WHERE dataset_id=? AND owner_id=? AND workspace_id=?""",
                (dataset_id, owner_id, workspace_id)).fetchone()
            if row is None:
                raise KeyError("Dataset not found")
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.update(updates)
            next_status = status or row["status"]
            conn.execute("UPDATE dl_datasets SET metadata_json=?, status=? WHERE dataset_id=?",
                         (json.dumps(metadata, sort_keys=True), next_status, dataset_id))
            updated = conn.execute("SELECT * FROM dl_datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        return self._public_dataset(dict(updated))

    def claim_job(self, job_id: str, *, owner_id: str, workspace_id: str) -> bool:
        """Atomically claim a queued job so duplicate dispatchers cannot train twice."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                "SELECT job_id FROM dl_training_jobs WHERE status='running' AND job_id<>? LIMIT 1",
                (job_id,),
            ).fetchone()
            if active is not None:
                return False
            cursor = conn.execute("""UPDATE dl_training_jobs
                SET status='running', stage='validating', progress=2, updated_at=?
                WHERE job_id=? AND owner_id=? AND workspace_id=? AND status='queued'""",
                (_now(), job_id, owner_id, workspace_id))
            return cursor.rowcount == 1

    def get_next_queued_job(self) -> Optional[dict[str, str]]:
        """Return the oldest queued job's dispatch identity (worker-internal use)."""
        with self.connect() as conn:
            row = conn.execute("""SELECT job_id,owner_id,workspace_id FROM dl_training_jobs
                WHERE status='queued' ORDER BY created_at ASC LIMIT 1""").fetchone()
        return dict(row) if row else None

    def update_job(self, job_id: str, *, owner_id: str, workspace_id: str,
                   status: str, stage: str, progress: float,
                   error_code: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM dl_training_jobs WHERE job_id=? AND owner_id=? AND workspace_id=?",
                               (job_id, owner_id, workspace_id)).fetchone()
            if row is None:
                raise KeyError("Training job not found")
            allowed_transitions = {
                "queued": {"running"},
                "running": {"running", "completed", "failed"},
            }
            current_status = row["status"]
            if status not in allowed_transitions.get(current_status, set()):
                raise ValueError(
                    f"Invalid training job transition: {current_status} -> {status}"
                )
            current = json.loads(row["metadata_json"] or "{}")
            current.update(metadata or {})
            conn.execute("""UPDATE dl_training_jobs SET status=?,stage=?,progress=?,error_code=?,
                updated_at=?,metadata_json=? WHERE job_id=? AND owner_id=? AND workspace_id=?""",
                (status, stage, max(0.0, min(100.0, float(progress))), error_code, _now(),
                 json.dumps(current, sort_keys=True), job_id, owner_id, workspace_id))
            updated = conn.execute("SELECT * FROM dl_training_jobs WHERE job_id=?", (job_id,)).fetchone()
        item = dict(updated)
        item.pop("metadata_json", None)
        return item

    def create_model(self, *, model_id: str, version: int, dataset_id: str, data_id: str,
                     owner_id: str, workspace_id: str, model_family: str, task_type: str,
                     artifact_path: str, metrics: dict[str, Any],
                     preprocessing: dict[str, Any], status: str = "candidate") -> dict[str, Any]:
        item = {"model_id": model_id, "version": int(version), "dataset_id": dataset_id,
                "data_id": data_id, "owner_id": owner_id, "workspace_id": workspace_id,
                "model_family": model_family, "task_type": task_type,
                "artifact_path": artifact_path, "metrics_json": json.dumps(metrics, sort_keys=True),
                "preprocessing_json": json.dumps(preprocessing, sort_keys=True),
                "status": status, "created_at": _now()}
        with self.connect() as conn:
            conn.execute("""INSERT INTO dl_models
                (model_id,version,dataset_id,data_id,owner_id,workspace_id,model_family,task_type,
                 artifact_path,metrics_json,preprocessing_json,status,created_at)
                VALUES (:model_id,:version,:dataset_id,:data_id,:owner_id,:workspace_id,:model_family,
                 :task_type,:artifact_path,:metrics_json,:preprocessing_json,:status,:created_at)""", item)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["preprocessing"] = json.loads(item.pop("preprocessing_json"))
        item.pop("artifact_path", None)
        return item

    def list_models(self, *, owner_id: str, workspace_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT * FROM dl_models
                WHERE owner_id=? AND workspace_id=? ORDER BY created_at DESC""",
                (owner_id, workspace_id)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["metrics"]=json.loads(item.pop("metrics_json"))
            item["preprocessing"]=json.loads(item.pop("preprocessing_json"))
            item.pop("artifact_path", None)
            result.append(item)
        return result


registry = DeepLearningRegistry()
