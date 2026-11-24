from __future__ import annotations

import json
import os
import socket
import sqlite3
from pathlib import Path

import src.deep_learning.registry as registry_module
from src.deep_learning.registry import DeepLearningRegistry


def _running_job(tmp_path: Path) -> tuple[DeepLearningRegistry, dict, dict]:
    registry = DeepLearningRegistry(tmp_path / "registry.sqlite3")
    dataset = registry.create_dataset(
        data_id=f"DATA-{os.urandom(4).hex()}",
        owner_id="owner",
        workspace_id="workspace",
        file_name="synthetic.csv",
        file_path=str(tmp_path / "synthetic.csv"),
        file_type="csv",
        size_bytes=1,
        sha256="a" * 64,
        classification="internal",
    )
    job = registry.approve_dataset(
        dataset["dataset_id"],
        owner_id="owner",
        workspace_id="workspace",
        device="cpu",
    )
    assert registry.claim_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    return registry, dataset, job


def _make_heartbeat_stale(registry: DeepLearningRegistry, job_id: str) -> None:
    with registry.connect() as conn:
        conn.execute(
            "UPDATE dl_training_jobs SET heartbeat_at=? WHERE job_id=?",
            ("2000-01-01T00:00:00+00:00", job_id),
        )


def test_claim_records_unique_worker_lease_and_heartbeat_is_token_guarded(tmp_path):
    registry, _dataset, job = _running_job(tmp_path)
    lease = registry.get_worker_lease(job["job_id"])

    assert lease["worker_token"]
    assert lease["worker_hostname"] == socket.gethostname()
    assert lease["worker_pid"] == os.getpid()
    assert lease["heartbeat_at"]
    assert registry.heartbeat_job(
        job["job_id"], worker_token=lease["worker_token"]
    )
    assert not registry.heartbeat_job(job["job_id"], worker_token="wrong-token")


def test_recovery_does_not_fail_a_live_worker_even_with_stale_heartbeat(
    tmp_path, monkeypatch
):
    registry, _dataset, job = _running_job(tmp_path)
    _make_heartbeat_stale(registry, job["job_id"])
    monkeypatch.setattr(registry_module, "_is_recorded_process_alive", lambda *_: True)

    assert registry.recover_stale_jobs(stale_after_seconds=1) == []
    current = registry.get_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    assert current["status"] == "running"


def test_recovery_marks_only_verified_dead_worker_failed(tmp_path, monkeypatch):
    registry, _dataset, job = _running_job(tmp_path)
    _make_heartbeat_stale(registry, job["job_id"])
    monkeypatch.setattr(registry_module, "_is_recorded_process_alive", lambda *_: False)

    assert registry.recover_stale_jobs(stale_after_seconds=1) == [job["job_id"]]
    current = registry.get_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    assert current["status"] == "failed"
    assert current["error_code"] == "WorkerProcessLost"
    with registry.connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM dl_training_jobs WHERE job_id=?",
            (job["job_id"],),
        ).fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["recovery"]["reason"] == "verified_worker_process_not_alive"
    assert registry.recover_stale_jobs(stale_after_seconds=1) == []


def test_recovery_skips_other_hosts_and_jobs_without_process_identity(
    tmp_path, monkeypatch
):
    registry, _dataset, job = _running_job(tmp_path)
    _make_heartbeat_stale(registry, job["job_id"])
    with registry.connect() as conn:
        conn.execute(
            "UPDATE dl_training_jobs SET worker_hostname=?,worker_process_started_at=NULL "
            "WHERE job_id=?",
            ("another-host", job["job_id"]),
        )
    monkeypatch.setattr(registry_module, "_is_recorded_process_alive", lambda *_: False)

    assert registry.recover_stale_jobs(stale_after_seconds=1) == []
    current = registry.get_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    assert current["status"] == "running"


def test_legacy_registry_schema_gets_additive_lease_migration(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE dl_training_jobs (
                job_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                data_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL, status TEXT NOT NULL,
                stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                selected_device TEXT NOT NULL, execution_environment TEXT NOT NULL,
                error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )"""
        )

    DeepLearningRegistry(db_path)
    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(dl_training_jobs)")
        }

    assert {
        "worker_token",
        "worker_hostname",
        "worker_pid",
        "worker_process_started_at",
        "heartbeat_at",
    }.issubset(columns)


def test_public_job_payload_does_not_expose_worker_lease_token(tmp_path):
    registry, _dataset, job = _running_job(tmp_path)
    public_job = registry.get_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    assert public_job is not None
    assert "worker_token" not in public_job
    assert "worker_pid" not in public_job
    assert "worker_process_started_at" not in public_job
    repeated_approval = registry.approve_dataset(
        job["dataset_id"], owner_id="owner", workspace_id="workspace", device="cpu"
    )
    assert "worker_token" not in repeated_approval

def test_recovery_cleans_only_the_orphan_artifact_for_the_dead_job(
    tmp_path, monkeypatch
):
    registry, dataset, job = _running_job(tmp_path)
    _make_heartbeat_stale(registry, job["job_id"])
    monkeypatch.setattr(
        registry_module, "_is_recorded_process_alive", lambda *_: False
    )

    data_root = tmp_path / "dataset-root"
    monkeypatch.setenv("EVA_DL_DATASET_DIR", str(data_root))
    artifact_dir = data_root / "processed" / dataset["dataset_id"]
    artifact_dir.mkdir(parents=True)

    model_id = "EVA-MODEL-" + "A" * 32
    artifact = artifact_dir / f"{model_id}_v1.json"
    artifact.write_text('{"synthetic": true}', encoding="utf-8")

    registry.record_artifact_intent(
        job["job_id"],
        owner_id="owner",
        workspace_id="workspace",
        dataset_id=dataset["dataset_id"],
        model_id=model_id,
        version=1,
        artifact_path=str(artifact),
    )

    assert artifact.exists()
    assert registry.recover_stale_jobs(stale_after_seconds=1) == [job["job_id"]]
    assert not artifact.exists()

    recovered = registry.get_job(
        job["job_id"], owner_id="owner", workspace_id="workspace"
    )
    with registry.connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM dl_training_jobs WHERE job_id=?",
            (job["job_id"],),
        ).fetchone()
    metadata = json.loads(row["metadata_json"])
    assert recovered["status"] == "failed"
    assert metadata["artifact_cleanup"]["status"] in {
        "removed", "removed_or_missing"
    }

    # A repeated recovery call is harmless and does not change the outcome.
    assert registry.recover_stale_jobs(stale_after_seconds=1) == []
    assert not artifact.exists()


def test_recovery_never_deletes_a_registered_model_artifact(
    tmp_path, monkeypatch
):
    registry, dataset, job = _running_job(tmp_path)
    _make_heartbeat_stale(registry, job["job_id"])
    monkeypatch.setattr(
        registry_module, "_is_recorded_process_alive", lambda *_: False
    )

    data_root = tmp_path / "dataset-root"
    monkeypatch.setenv("EVA_DL_DATASET_DIR", str(data_root))
    artifact_dir = data_root / "processed" / dataset["dataset_id"]
    artifact_dir.mkdir(parents=True)

    model_id = "EVA-MODEL-" + "B" * 32
    artifact = artifact_dir / f"{model_id}_v1.json"
    artifact.write_text('{"registered": true}', encoding="utf-8")

    registry.record_artifact_intent(
        job["job_id"],
        owner_id="owner",
        workspace_id="workspace",
        dataset_id=dataset["dataset_id"],
        model_id=model_id,
        version=1,
        artifact_path=str(artifact),
    )
    registry.create_model(
        model_id=model_id,
        version=1,
        dataset_id=dataset["dataset_id"],
        data_id=dataset["data_id"],
        owner_id="owner",
        workspace_id="workspace",
        model_family="synthetic",
        task_type="classification",
        artifact_path=str(artifact),
        metrics={"accuracy": 1.0},
        preprocessing={},
    )

    assert registry.recover_stale_jobs(stale_after_seconds=1) == [job["job_id"]]
    assert artifact.exists()

    with registry.connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM dl_training_jobs WHERE job_id=?",
            (job["job_id"],),
        ).fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["artifact_cleanup"]["status"] == "skipped_registered"
