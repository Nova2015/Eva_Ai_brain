"""SQLite-backed dataset, training-job and model metadata registry."""
from __future__ import annotations

import json
import os
import sqlite3
import socket
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from .config import get_dataset_root, get_registry_db_path





def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_eva_id(value: Any, prefix: str) -> bool:
    """Validate EVA IDs with a fixed prefix and 32 uppercase hex characters."""
    return (
        isinstance(value, str)
        and len(value) == len(prefix) + 32
        and value.startswith(prefix)
        and all(c in "0123456789ABCDEF" for c in value[len(prefix):])
    )


def _is_unsafe_path_entry(path: Path) -> bool:
    """Return True for symlinks, junctions, or paths that cannot be safely checked."""
    try:
        if path.is_symlink():
            return True

        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True

        return False
    except (OSError, RuntimeError, ValueError):
        # Fail closed: an uncheckable filesystem entry must never be deleted.
        return True


def _process_start_time(pid: int) -> float | None:
    """Return OS process creation time; None means liveness cannot be verified."""
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _is_recorded_process_alive(pid: int, expected_start_time: float) -> bool | None:
    """True=recorded process alive, False=dead/PID reused, None=unknown."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        actual_start_time = float(process.create_time())
        return abs(actual_start_time - expected_start_time) <= 1.5
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, psutil.ZombieProcess):
        return None
    except Exception:
        return None


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


class DeepLearningRegistry:
    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = Path(
            db_path if db_path is not None else get_registry_db_path()
            ).expanduser()
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
                    worker_token TEXT, worker_hostname TEXT, worker_pid INTEGER,
                    worker_process_started_at REAL, heartbeat_at TEXT,
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
                CREATE TABLE IF NOT EXISTS dl_artifact_cleanup_claims (
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'claimed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT,
                    PRIMARY KEY(model_id, version)
                );
            """)
            # Additive, idempotent migration for databases created by older builds.
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(dl_training_jobs)").fetchall()
            }
            lease_columns = {
                "worker_token": "TEXT",
                "worker_hostname": "TEXT",
                "worker_pid": "INTEGER",
                "worker_process_started_at": "REAL",
                "heartbeat_at": "TEXT",
            }
            for column_name, column_type in lease_columns.items():
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE dl_training_jobs ADD COLUMN {column_name} {column_type}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dl_jobs_worker_heartbeat "
                "ON dl_training_jobs(status, heartbeat_at)"
            )

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
                return self._public_job(dict(existing))
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

    @staticmethod
    def _public_job(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item.pop("metadata_json", None)
        for internal_field in (
            "worker_token", "worker_hostname", "worker_pid",
            "worker_process_started_at", "heartbeat_at",
        ):
            item.pop(internal_field, None)
        return item

    def get_job(self, job_id: str, *, owner_id: str, workspace_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM dl_training_jobs
                WHERE job_id=? AND owner_id=? AND workspace_id=?""",
                (job_id, owner_id, workspace_id)).fetchone()
        if not row:
            return None
        return self._public_job(dict(row))

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

    def claim_job(self, job_id: str, *, owner_id: str, workspace_id: str,
                  worker_token: str | None = None) -> bool:
        """Atomically claim a queued job and bind it to this OS worker process."""
        token = worker_token or uuid.uuid4().hex
        pid = os.getpid()
        process_started_at = _process_start_time(pid)
        now = _now()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                "SELECT job_id FROM dl_training_jobs WHERE status='running' AND job_id<>? LIMIT 1",
                (job_id,),
            ).fetchone()
            if active is not None:
                return False
            cursor = conn.execute(
                """UPDATE dl_training_jobs
                   SET status='running', stage='validating', progress=2, updated_at=?,
                       worker_token=?, worker_hostname=?, worker_pid=?,
                       worker_process_started_at=?, heartbeat_at=?
                   WHERE job_id=? AND owner_id=? AND workspace_id=? AND status='queued'""",
                (now, token, socket.gethostname(), pid, process_started_at, now,
                 job_id, owner_id, workspace_id),
            )
            return cursor.rowcount == 1

    def get_worker_lease(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return worker lease details for internal worker code only."""
        with self.connect() as conn:
            row = conn.execute(
                """SELECT worker_token, worker_hostname, worker_pid,
                          worker_process_started_at, heartbeat_at, status
                   FROM dl_training_jobs WHERE job_id=?""",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def heartbeat_job(self, job_id: str, *, worker_token: str) -> bool:
        """Refresh a running job lease only when the caller still owns its token."""
        now = _now()
        with self.connect() as conn:
            cursor = conn.execute(
                """UPDATE dl_training_jobs SET heartbeat_at=?, updated_at=?
                   WHERE job_id=? AND status='running' AND worker_token=?""",
                (now, now, job_id, worker_token),
            )
            return cursor.rowcount == 1


    def record_artifact_intent(
        self, job_id: str, *, owner_id: str, workspace_id: str,
        dataset_id: str, model_id: str, version: int, artifact_path: str,
    ) -> None:
        """Persist the exact artifact target atomically before the worker writes it."""
        if not _is_valid_eva_id(model_id, "EVA-MODEL-"):
            raise ValueError("Invalid model ID for artifact intent")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Artifact version must be a positive integer")
        if not _is_valid_eva_id(dataset_id, "EVA-DSET-"):
            raise ValueError("Invalid dataset ID for artifact intent")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError("Invalid artifact path")

        expected_name = f"{model_id}_v{version}.json"
        try:
            supplied = Path(artifact_path).expanduser().resolve(strict=False)
            root = get_dataset_root().resolve(strict=False)
            expected = (
                root / "processed" / dataset_id / expected_name
            ).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("Unable to validate artifact path") from exc

        if supplied != expected:
            raise ValueError(
                "Artifact path does not match the expected dataset artifact location"
            )

        with self.connect() as conn:
            # Serialize intent recording against stale-job recovery.
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                """SELECT dataset_id, metadata_json, status
                   FROM dl_training_jobs
                   WHERE job_id=? AND owner_id=? AND workspace_id=?""",
                (job_id, owner_id, workspace_id),
            ).fetchone()

            if row is None:
                raise KeyError("Training job not found")

            if row["status"] != "running" or row["dataset_id"] != dataset_id:
                raise ValueError("Artifact intent does not match the running job")

            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["artifact_intent"] = {
                "model_id": model_id,
                "version": version,
                "dataset_id": dataset_id,
                "artifact_path": str(supplied),
            }

            cursor = conn.execute(
                """UPDATE dl_training_jobs
                   SET metadata_json=?, updated_at=?
                   WHERE job_id=? AND owner_id=? AND workspace_id=?
                     AND status='running' AND dataset_id=?""",
                (
                    json.dumps(metadata, sort_keys=True),
                    _now(),
                    job_id,
                    owner_id,
                    workspace_id,
                    dataset_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    "Artifact intent could not be persisted; refusing artifact creation"
                )


    def _set_artifact_cleanup_state(
        self, job_id: str, state: str, *, error: str | None = None,
    ) -> None:
        now = _now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM dl_training_jobs WHERE job_id=? AND status='failed'",
                (job_id,),
            ).fetchone()
            if row is None:
                return

            metadata = json.loads(row["metadata_json"] or "{}")
            cleanup = metadata.get("artifact_cleanup") or {}
            cleanup["status"] = state
            cleanup["updated_at"] = now
            if error:
                cleanup["last_error"] = error
            else:
                cleanup.pop("last_error", None)
            metadata["artifact_cleanup"] = cleanup

            conn.execute(
                """UPDATE dl_training_jobs
                   SET metadata_json=?, updated_at=?
                   WHERE job_id=? AND status='failed'""",
                (json.dumps(metadata, sort_keys=True), now, job_id),
            )
            conn.execute(
                """UPDATE dl_artifact_cleanup_claims
                   SET status=?, updated_at=?, last_error=?
                   WHERE job_id=?""",
                (state, now, error, job_id),
            )

    def cleanup_failed_job_artifact(
        self, job_id: str, *, require_verified_recovery: bool = False,
    ) -> str:
        """Safely clean one failed job's exact artifact using an atomic DB claim.

        A durable claim serializes cleanup against create_model(). Filesystem
        deletion happens after the claim transaction commits. A failed unlink
        leaves the claim in place so only this same job can retry it.
        """
        with self.connect() as conn:
            row = conn.execute(
                """SELECT job_id, dataset_id, status, error_code, metadata_json
                   FROM dl_training_jobs WHERE job_id=?""",
                (job_id,),
            ).fetchone()

        if row is None or row["status"] != "failed":
            return "not_eligible"

        metadata = json.loads(row["metadata_json"] or "{}")
        recovery = metadata.get("recovery") or {}
        intent = metadata.get("artifact_intent") or {}
        cleanup = metadata.get("artifact_cleanup") or {}

        if require_verified_recovery:
            if (
                row["error_code"] != "WorkerProcessLost"
                or recovery.get("reason") != "verified_worker_process_not_alive"
            ):
                return "not_eligible"

        if cleanup.get("status") not in {"pending", "retry"}:
            return str(cleanup.get("status") or "not_pending")

        model_id = intent.get("model_id")
        version = intent.get("version")
        dataset_id = intent.get("dataset_id")
        recorded_path = intent.get("artifact_path")

        if (
            not _is_valid_eva_id(model_id, "EVA-MODEL-")
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or not _is_valid_eva_id(dataset_id, "EVA-DSET-")
            or dataset_id != row["dataset_id"]
            or not isinstance(recorded_path, str)
            or not recorded_path
        ):
            self._set_artifact_cleanup_state(job_id, "blocked_invalid_intent")
            return "blocked_invalid_intent"

        # Resolve the exact permitted target and reject symlinked path components.
        try:
            root = get_dataset_root().resolve(strict=False)

            processed_dir = root / "processed"
            dataset_dir = processed_dir / dataset_id
            expected = dataset_dir / f"{model_id}_v{version}.json"

            if (
                _is_unsafe_path_entry(processed_dir)
                or _is_unsafe_path_entry(dataset_dir)
                or _is_unsafe_path_entry(expected)
            ):
                self._set_artifact_cleanup_state(job_id, "blocked_unsafe_path")
                return "blocked_unsafe_path"

            expected_resolved = expected.resolve(strict=False)
            dataset_resolved = dataset_dir.resolve(strict=False)
            recorded_resolved = Path(recorded_path).expanduser().resolve(strict=False)

            if (
                recorded_resolved != expected_resolved
                or expected_resolved.parent != dataset_resolved
                or expected_resolved.name != f"{model_id}_v{version}.json"
            ):
                self._set_artifact_cleanup_state(job_id, "blocked_unsafe_path")
                return "blocked_unsafe_path"
        except (OSError, RuntimeError, ValueError):
            self._set_artifact_cleanup_state(job_id, "blocked_unsafe_path")
            return "blocked_unsafe_path"

        now = _now()

        # Atomic serialization point: create_model() uses the same write lock.
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """SELECT dataset_id, status, error_code, metadata_json
                   FROM dl_training_jobs WHERE job_id=?""",
                (job_id,),
            ).fetchone()

            if current is None or current["status"] != "failed":
                return "not_eligible"

            current_metadata = json.loads(current["metadata_json"] or "{}")
            current_intent = current_metadata.get("artifact_intent") or {}
            current_cleanup = current_metadata.get("artifact_cleanup") or {}

            if (
                current["dataset_id"] != dataset_id
                or current_intent != intent
                or current_cleanup.get("status") not in {"pending", "retry"}
            ):
                return "intent_changed"

            if require_verified_recovery:
                current_recovery = current_metadata.get("recovery") or {}
                if (
                    current["error_code"] != "WorkerProcessLost"
                    or current_recovery.get("reason")
                    != "verified_worker_process_not_alive"
                ):
                    return "not_eligible"

            registered = conn.execute(
                """SELECT 1 FROM dl_models
                   WHERE model_id=? AND version=? LIMIT 1""",
                (model_id, version),
            ).fetchone()

            if registered:
                # A registered artifact is never removed.
                current_cleanup["status"] = "skipped_registered"
                current_cleanup["updated_at"] = now
                current_metadata["artifact_cleanup"] = current_cleanup
                conn.execute(
                    """UPDATE dl_training_jobs
                       SET metadata_json=?, updated_at=?
                       WHERE job_id=? AND status='failed'""",
                    (json.dumps(current_metadata, sort_keys=True), now, job_id),
                )
                return "skipped_registered"

            existing_claim = conn.execute(
                """SELECT job_id, dataset_id, artifact_path
                   FROM dl_artifact_cleanup_claims
                   WHERE model_id=? AND version=?""",
                (model_id, version),
            ).fetchone()

            if existing_claim:
                if (
                    existing_claim["job_id"] != job_id
                    or existing_claim["dataset_id"] != dataset_id
                    or existing_claim["artifact_path"] != str(expected_resolved)
                ):
                    current_cleanup["status"] = "blocked_existing_claim"
                    current_cleanup["updated_at"] = now
                    current_metadata["artifact_cleanup"] = current_cleanup
                    conn.execute(
                        """UPDATE dl_training_jobs
                           SET metadata_json=?, updated_at=?
                           WHERE job_id=? AND status='failed'""",
                        (json.dumps(current_metadata, sort_keys=True), now, job_id),
                    )
                    return "blocked_existing_claim"
            else:
                conn.execute(
                    """INSERT INTO dl_artifact_cleanup_claims
                       (model_id, version, job_id, dataset_id, artifact_path,
                        status, created_at, updated_at, last_error)
                       VALUES (?, ?, ?, ?, ?, 'claimed', ?, ?, NULL)""",
                    (
                        model_id, version, job_id, dataset_id,
                        str(expected_resolved), now, now,
                    ),
                )

        # The durable claim is committed. Recheck symlink state immediately
        # before unlinking; do not hold a SQLite transaction during I/O.
        try:
            if (
                _is_unsafe_path_entry(processed_dir)
                or _is_unsafe_path_entry(dataset_dir)
                or _is_unsafe_path_entry(expected)
            ):
                self._set_artifact_cleanup_state(job_id, "blocked_unsafe_path")
                return "blocked_unsafe_path"

            artifact_existed = expected.exists()
            expected.unlink(missing_ok=True)
            state = "removed" if artifact_existed else "removed_or_missing"
            self._set_artifact_cleanup_state(job_id, state)
            return state

        except (RuntimeError, ValueError):
            self._set_artifact_cleanup_state(job_id, "blocked_unsafe_path")
            return "blocked_unsafe_path"
        except OSError as exc:
            self._set_artifact_cleanup_state(
                job_id, "pending", error=type(exc).__name__
            )
            return "pending"

    def cleanup_pending_artifacts(self) -> None:
        """Retry cleanup only for recovered jobs with verified dead processes."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT job_id
                   FROM dl_training_jobs
                   WHERE status='failed' AND error_code='WorkerProcessLost'"""
            ).fetchall()

        for row in rows:
            try:
                self.cleanup_failed_job_artifact(
                    row["job_id"], require_verified_recovery=True
                )
            except Exception:
                # One damaged row or transient DB error must not stop other jobs.
                continue

    def recover_stale_jobs(self, *, stale_after_seconds: float = 60.0) -> list[str]:
        """Fail only stale jobs whose recorded local OS process is proven dead.

        Jobs from another host, legacy jobs without process identity, and jobs
        whose process state cannot be verified are deliberately left untouched.
        Jobs are never automatically re-queued.
        """
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be non-negative")
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=float(stale_after_seconds))
        recovered: list[str] = []
        local_hostname = socket.gethostname()

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT job_id, worker_token, worker_hostname, worker_pid,
                          worker_process_started_at, heartbeat_at, metadata_json
                   FROM dl_training_jobs
                   WHERE status='running' AND worker_token IS NOT NULL"""
            ).fetchall()

            for row in rows:
                heartbeat = _parse_utc(row["heartbeat_at"])
                if heartbeat is None or heartbeat > cutoff:
                    continue
                if row["worker_hostname"] != local_hostname:
                    continue
                if row["worker_pid"] is None or row["worker_process_started_at"] is None:
                    continue
                process_alive = _is_recorded_process_alive(
                    int(row["worker_pid"]), float(row["worker_process_started_at"])
                )
                if process_alive is not False:
                    continue

                metadata = json.loads(row["metadata_json"] or "{}")
                metadata.update({
                    "failure_message": "Training worker process was no longer running; job marked failed during recovery.",
                    "recovery": {
                        "reason": "verified_worker_process_not_alive",
                        "recovered_at": _now(),
                        "worker_hostname": row["worker_hostname"],
                        "worker_pid": row["worker_pid"],
                        "last_heartbeat_at": row["heartbeat_at"],
                    },
                })
                if metadata.get("artifact_intent"):
                    metadata["artifact_cleanup"] = {
                        "status": "pending",
                        "created_at": _now(),
                    }
                cursor = conn.execute(
                    """UPDATE dl_training_jobs
                       SET status='failed', stage='failed', progress=0,
                           error_code='WorkerProcessLost', updated_at=?, metadata_json=?
                       WHERE job_id=? AND status='running' AND worker_token=?""",
                    (_now(), json.dumps(metadata, sort_keys=True),
                     row["job_id"], row["worker_token"]),
                )
                if cursor.rowcount == 1:
                    recovered.append(row["job_id"])
        # The SQLite recovery transaction has committed before filesystem cleanup.
        # Also retries any earlier cleanup that remained pending.
        self.cleanup_pending_artifacts()
        return recovered

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
        return self._public_job(dict(updated))

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
            conn.execute("BEGIN IMMEDIATE")
            claim = conn.execute(
                """SELECT job_id FROM dl_artifact_cleanup_claims
                   WHERE model_id=? AND version=? LIMIT 1""",
                (model_id, item["version"]),
            ).fetchone()
            if claim:
                raise RuntimeError(
                    "Model artifact cleanup has claimed this model identity; "
                    "registration is blocked."
                )
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