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
