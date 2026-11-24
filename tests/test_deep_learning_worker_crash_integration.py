from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.deep_learning.registry import DeepLearningRegistry


CHILD_SCRIPT = r"""
import json
import os
import sys
import time
from pathlib import Path

from src.deep_learning.registry import DeepLearningRegistry
from src.deep_learning.worker import _WorkerHeartbeat

job_id, dataset_id, ready_path = sys.argv[1:4]
registry = DeepLearningRegistry()

if not registry.claim_job(
    job_id, owner_id="crash-test-owner",
    workspace_id="crash-test-workspace"
):
    raise SystemExit("Worker could not claim synthetic job")

lease = registry.get_worker_lease(job_id)
heartbeat = _WorkerHeartbeat(job_id, lease["worker_token"], registry)
heartbeat.start()

# Persist the exact artifact intent before writing the artifact.
model_id = "EVA-MODEL-" + ("C" * 32)
dataset_root = Path(os.environ["EVA_DL_DATASET_DIR"])
artifact_dir = dataset_root / "processed" / dataset_id
artifact_path = artifact_dir / f"{model_id}_v1.json"

registry.record_artifact_intent(
    job_id,
    owner_id="crash-test-owner",
    workspace_id="crash-test-workspace",
    dataset_id=dataset_id,
    model_id=model_id,
    version=1,
    artifact_path=str(artifact_path),
)

artifact_dir.mkdir(parents=True, exist_ok=True)
artifact_path.write_text(
    json.dumps({"synthetic": True, "model_id": model_id}),
    encoding="utf-8",
)

Path(ready_path).write_text(str(os.getpid()), encoding="utf-8")

while True:
    time.sleep(0.2)
"""


def test_real_worker_process_crash_is_recovered(tmp_path, monkeypatch):
    db_path = tmp_path / "crash-recovery.sqlite3"
    ready_file = tmp_path / "worker-ready.txt"
    dataset_root = tmp_path / "dataset-root"
    model_id = "EVA-MODEL-" + ("C" * 32)

    monkeypatch.setenv("EVA_DL_REGISTRY_DB", str(db_path))
    monkeypatch.setenv("EVA_DL_DATASET_DIR", str(dataset_root))
    monkeypatch.setenv(
        "EVA_DL_WORKER_HEARTBEAT_INTERVAL_SECONDS", "1"
    )

    registry = DeepLearningRegistry(db_path)

    dataset = registry.create_dataset(
        data_id="EVA-DATA-REAL-CRASH-TEST",
        owner_id="crash-test-owner",
        workspace_id="crash-test-workspace",
        file_name="synthetic.csv",
        file_path=str(tmp_path / "synthetic.csv"),
        file_type=".csv",
        size_bytes=10,
        sha256="a" * 64,
        classification="internal",
    )

    job = registry.approve_dataset(
        dataset["dataset_id"],
        owner_id="crash-test-owner",
        workspace_id="crash-test-workspace",
        device="cpu",
        environment="local",
    )

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["EVA_DL_REGISTRY_DB"] = str(db_path)
    env["EVA_DL_WORKER_HEARTBEAT_INTERVAL_SECONDS"] = "1"
    env["PYTHONPATH"] = (
        str(project_root)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD_SCRIPT,
            job["job_id"],
            dataset["dataset_id"],
            str(ready_file),
        ],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if ready_file.exists():
                break
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"Worker exited before claiming job.\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.1)

        assert ready_file.exists(), "Worker did not become ready"

        lease = registry.get_worker_lease(job["job_id"])
        assert lease["worker_pid"] == process.pid
        assert lease["worker_token"]

        initial_heartbeat = lease["heartbeat_at"]
        deadline = time.monotonic() + 8

        while time.monotonic() < deadline:
            current = registry.get_worker_lease(job["job_id"])
            if current["heartbeat_at"] != initial_heartbeat:
                break
            time.sleep(0.1)

        assert current["heartbeat_at"] != initial_heartbeat, (
            "Child worker heartbeat did not refresh"
        )

        process.kill()
        process.wait(timeout=10)

        # Simulate an expired heartbeat after the real process has died.
        with registry.connect() as conn:
            conn.execute(
                """
                UPDATE dl_training_jobs
                SET heartbeat_at=?
                WHERE job_id=?
                """,
                ("2000-01-01T00:00:00+00:00", job["job_id"]),
            )

        recovered = registry.recover_stale_jobs(
            stale_after_seconds=1
        )
        assert recovered == [job["job_id"]]

        persisted = registry.get_job(
            job["job_id"],
            owner_id="crash-test-owner",
            workspace_id="crash-test-workspace",
        )
        assert persisted["status"] == "failed"
        assert persisted["error_code"] == "WorkerProcessLost"

        with registry.connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM dl_training_jobs WHERE job_id=?",
                (job["job_id"],),
            ).fetchone()

        metadata = json.loads(row["metadata_json"])
        artifact_path = (
            dataset_root
            / "processed"
            / dataset["dataset_id"]
            / f"{model_id}_v1.json"
        )

        assert not artifact_path.exists(), (
            "Orphan artifact was not removed after worker process crash"
        )
        assert metadata["artifact_cleanup"]["status"] == "removed"
        assert (
            metadata["recovery"]["reason"]
            == "verified_worker_process_not_alive"
        )

        # Recovery must be idempotent.
        assert registry.recover_stale_jobs(
            stale_after_seconds=1
        ) == []

    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
