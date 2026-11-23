import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import job_manager as jm
from src.rag import upload_policy


import pytest


@pytest.fixture(autouse=True)
def isolated_job_database(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_RAG_JOB_DB", str(tmp_path / "rag_jobs.sqlite3"))
    jm._reset_for_tests()
    yield
    jm._reset_for_tests()


def test_upload_limits_payload_is_backend_driven():
    os.environ["EVA_RUNTIME_MODE"] = "local"
    payload = upload_policy.limits_payload()
    assert payload["async_jobs"] is True
    assert payload["max_upload_mb"] >= 1
    assert payload["max_chunks"] >= 1


def test_job_manager_tracks_background_upload_state():
    job_id = "job-123"
    jm.create_job(job_id, filename="demo.txt")
    jm.update_job(job_id, status=jm.PARSING, progress=20, current_step="Parsing")
    job = jm.get_job(job_id)
    assert job["status"] == jm.PARSING
    assert job["progress"] == 20
    assert job["current_step"] == "Parsing"



def test_upload_job_is_visible_only_to_its_owner():
    job_id = "owner-bound-job"
    jm.create_job(job_id, filename="private.txt", owner_id="user-a", workspace_id="workspace-a")

    owner_view = jm.get_job(job_id, owner_id="user-a")
    other_user_view = jm.get_job(job_id, owner_id="user-b")

    assert owner_view["status"] == jm.QUEUED
    assert other_user_view["status"] == "NOT_FOUND"


def test_legacy_job_without_owner_is_not_visible_to_named_user():
    job_id = "legacy-anonymous-job"
    jm.create_job(job_id, filename="legacy.txt")

    assert jm.get_job(job_id, owner_id="user-a")["status"] == "NOT_FOUND"
    assert jm.get_job(job_id, owner_id="anonymous")["status"] == jm.QUEUED



def test_job_state_survives_cache_reset_and_keeps_owner_scope():
    jm.create_job("persisted-job", filename="model.pdf", owner_id="user-a", workspace_id="cad")
    jm.update_job(
        "persisted-job",
        status=jm.INDEXING,
        progress=70,
        data_id="EVA-DATA-123",
        document_path="/private/local/model.pdf",
    )

    # Simulate a fresh process: only the SQLite record remains.
    jm._reset_for_tests()
    owner_view = jm.get_job("persisted-job", owner_id="user-a")
    other_view = jm.get_job("persisted-job", owner_id="user-b")

    assert owner_view["status"] == jm.INDEXING
    assert owner_view["data_id"] == "EVA-DATA-123"
    assert "document_path" not in owner_view
    assert other_view["status"] == "NOT_FOUND"


def test_restart_recovery_marks_active_jobs_failed_without_losing_source_reference():
    jm.create_job("interrupted-job", filename="source.step", owner_id="user-a", workspace_id="cad")
    jm.update_job(
        "interrupted-job",
        status=jm.INDEXING,
        progress=70,
        data_id="EVA-DATA-456",
        document_path="/private/local/source.step",
    )
    jm._reset_for_tests()

    recovered = jm.recover_interrupted_jobs()
    job = jm.get_job("interrupted-job", owner_id="user-a")

    assert recovered == 1
    assert job["status"] == jm.FAILED
    assert job["error"].startswith("PROCESS_RESTART")
    assert job["data_id"] == "EVA-DATA-456"


def test_finished_job_cleanup_does_not_delete_governed_source_file(tmp_path):
    source = tmp_path / "governed.txt"
    source.write_text("keep this governed source", encoding="utf-8")
    jm.create_job("finished-job", filename="governed.txt")
    jm.update_job("finished-job", status=jm.COMPLETE, progress=100)
    jm.update_job("finished-job", document_path=str(source))
    removed = jm.cleanup_finished_jobs(max_age_seconds=0)

    assert removed == 1
    assert source.exists()
