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



def test_retry_job_links_to_original_and_preserves_private_source_reference():
    jm.create_job(
        "original-failed",
        filename="assembly.pdf",
        owner_id="user-a",
        workspace_id="cad",
    )
    jm.update_job(
        "original-failed",
        status=jm.FAILED,
        data_id="EVA-DATA-789",
        document_path="/private/governed/assembly.pdf",
    )

    jm.create_job(
        "retry-created",
        filename="assembly.pdf",
        owner_id="user-a",
        workspace_id="cad",
        retry_of="original-failed",
    )
    internal = jm.get_job_internal("original-failed", owner_id="user-a")
    public_retry = jm.get_job("retry-created", owner_id="user-a")

    assert internal["data_id"] == "EVA-DATA-789"
    assert internal["document_path"] == "/private/governed/assembly.pdf"
    assert public_retry["retry_of"] == "original-failed"
    assert "document_path" not in public_retry
    assert jm.get_job_internal("original-failed", owner_id="user-b") is None



def test_reconciliation_reports_orphans_missing_files_and_hash_mismatches(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))

    from src.data_governance import register_import
    from src.rag import hybrid_rag_v2 as rag

    def register(name, contents):
        source = tmp_path / name
        source.write_text(contents, encoding="utf-8")
        return register_import(
            source,
            source="test",
            owner_id="user-a",
            workspace_id="cad",
            actor_id="user-a",
        ), source

    indexed_record, _ = register("indexed.txt", "indexed source")
    missing_index_record, _ = register("missing-index.txt", "not indexed")
    changed_record, changed_path = register("changed.txt", "original bytes")
    missing_file_record, missing_file_path = register("missing.txt", "will be removed")
    missing_file_path.unlink()
    changed_path.write_text("modified bytes", encoding="utf-8")

    monkeypatch.setattr(rag, "_chroma_chunks_for_user", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag,
        "_vectorless_chunks_for_user",
        lambda *args, **kwargs: [
            {"metadata": {"data_id": indexed_record["data_id"]}},
            {"metadata": {"data_id": changed_record["data_id"]}},
            {"metadata": {"data_id": "EVA-DATA-ORPHAN"}},
        ],
    )

    report = rag.reconcile_data_sources("user-a", workspace_id="cad")

    assert report["read_only"] is True
    assert report["summary"]["missing_from_index"] == 2
    assert report["summary"]["index_without_governance"] == 1
    assert report["summary"]["missing_files"] == 1
    assert report["summary"]["hash_mismatches"] == 1
    assert {item["data_id"] for item in report["missing_from_index"]} == {
        missing_index_record["data_id"],
        missing_file_record["data_id"],
    }
    assert report["hash_mismatches"][0]["data_id"] == changed_record["data_id"]
    assert report["actions_taken"] == []
    assert not missing_file_path.exists()
