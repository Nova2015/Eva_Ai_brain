import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import job_manager as jm
from src.rag import upload_policy


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
