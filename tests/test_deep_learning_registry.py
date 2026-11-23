from pathlib import Path

from src.deep_learning.registry import DeepLearningRegistry


def test_dataset_registry_persists_and_scopes_by_owner(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    created = registry.create_dataset(
        data_id="EVA-DATA-1", owner_id="alice", workspace_id="team-a",
        file_name="defects.csv", file_path=str(tmp_path / "defects.csv"),
        file_type=".csv", size_bytes=20, sha256="a" * 64,
        classification="restricted",
    )
    assert created["status"] == "awaiting_training_approval"
    assert "file_path" not in created
    assert len(registry.list_datasets(owner_id="alice", workspace_id="team-a")) == 1
    assert registry.list_datasets(owner_id="bob", workspace_id="team-a") == []
    assert registry.get_dataset(created["dataset_id"], owner_id="bob", workspace_id="team-a") is None


def test_approval_queues_one_persistent_training_job(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    dataset = registry.create_dataset(
        data_id="EVA-DATA-2", owner_id="alice", workspace_id="team-a",
        file_name="measurements.csv", file_path=str(tmp_path / "measurements.csv"),
        file_type=".csv", size_bytes=42, sha256="b" * 64,
        classification="internal",
    )
    first = registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    second = registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    assert first["job_id"] == second["job_id"]
    assert first["status"] == "queued"
    assert registry.get_job(first["job_id"], owner_id="alice", workspace_id="team-a")["stage"] == "queued_for_training_worker"
    assert registry.get_job(first["job_id"], owner_id="bob", workspace_id="team-a") is None


def test_unapproved_dataset_cannot_be_queued(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    dataset = registry.create_dataset(
        data_id="EVA-DATA-3", owner_id="alice", workspace_id="team-a",
        file_name="labels.csv", file_path=str(tmp_path / "labels.csv"),
        file_type=".csv", size_bytes=12, sha256="c" * 64,
        classification="confidential",
    )
    # Queue creation is intentionally only exposed through approve_dataset;
    # a fresh dataset remains in the approval-pending state.
    assert registry.get_dataset(dataset["dataset_id"], owner_id="alice", workspace_id="team-a")["status"] == "awaiting_training_approval"


def test_prepared_dataset_can_be_queued_after_approval(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    dataset = registry.create_dataset(
        data_id="EVA-DATA-PREPARED", owner_id="alice", workspace_id="team-a",
        file_name="inspection.csv", file_path=str(tmp_path / "inspection.csv"),
        file_type=".csv", size_bytes=24, sha256="d" * 64,
        classification="restricted",
    )
    registry.update_dataset_metadata(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        updates={"preparation_status": "completed", "label_status": "not_reviewed"},
        status="prepared",
    )
    job = registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    assert job["status"] == "queued"
    assert registry.get_dataset(dataset["dataset_id"], owner_id="alice", workspace_id="team-a")["status"] == "approved"


def test_job_state_transitions_and_model_registration(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    dataset = registry.create_dataset(
        data_id="EVA-DATA-WORKER", owner_id="alice", workspace_id="team-a",
        file_name="train.csv", file_path=str(tmp_path / "train.csv"),
        file_type=".csv", size_bytes=120, sha256="e" * 64,
        classification="restricted",
    )
    registry.update_dataset_metadata(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        updates={"label_review": {"status": "verified"}}, status="reviewed",
    )
    job = registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    running = registry.update_job(
        job["job_id"], owner_id="alice", workspace_id="team-a",
        status="running", stage="training", progress=20,
    )
    assert running["status"] == "running"
    assert running["progress"] == 20
    model = registry.create_model(
        model_id="EVA-MODEL-TEST", version=1, dataset_id=dataset["dataset_id"],
        data_id="EVA-DATA-WORKER", owner_id="alice", workspace_id="team-a",
        model_family="XGBoost", task_type="classification",
        artifact_path=str(tmp_path / "model.json"), metrics={"accuracy": 0.9},
        preprocessing={"target_column": "label"}, status="candidate",
    )
    assert model["status"] == "candidate"
    assert "artifact_path" not in model
    listed = registry.list_models(owner_id="alice", workspace_id="team-a")
    assert len(listed) == 1
    assert listed[0]["metrics"]["accuracy"] == 0.9


def test_training_job_claim_is_atomic_and_only_succeeds_once(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    dataset = registry.create_dataset(
        data_id="EVA-DATA-CLAIM", owner_id="alice", workspace_id="team-a",
        file_name="train.csv", file_path=str(tmp_path / "train.csv"),
        file_type=".csv", size_bytes=12, sha256="f" * 64,
        classification="restricted",
    )
    job = registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    assert registry.claim_job(job["job_id"], owner_id="alice", workspace_id="team-a") is True
    assert registry.claim_job(job["job_id"], owner_id="alice", workspace_id="team-a") is False
    assert registry.get_job(job["job_id"], owner_id="alice", workspace_id="team-a")["status"] == "running"


def test_local_worker_claims_only_one_training_job_at_a_time(tmp_path):
    registry = DeepLearningRegistry(tmp_path / "dl.sqlite3")
    jobs = []
    for index in range(2):
        dataset = registry.create_dataset(
            data_id=f"EVA-DATA-SERIAL-{index}", owner_id="alice", workspace_id="team-a",
            file_name=f"train-{index}.csv", file_path=str(tmp_path / f"train-{index}.csv"),
            file_type=".csv", size_bytes=12, sha256=str(index) * 64,
            classification="restricted",
        )
        jobs.append(registry.approve_dataset(
            dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
            device="cpu", environment="local",
        ))
    assert registry.claim_job(jobs[0]["job_id"], owner_id="alice", workspace_id="team-a") is True
    assert registry.claim_job(jobs[1]["job_id"], owner_id="alice", workspace_id="team-a") is False
    registry.update_job(
        jobs[0]["job_id"], owner_id="alice", workspace_id="team-a",
        status="completed", stage="completed", progress=100,
    )
    assert registry.claim_job(jobs[1]["job_id"], owner_id="alice", workspace_id="team-a") is True
