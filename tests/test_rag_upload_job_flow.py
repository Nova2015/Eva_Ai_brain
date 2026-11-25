import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import job_manager as jm
from src.rag import upload_policy
from src.data_governance import register_import


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











def test_local_graph_extractor_parses_json_and_marks_candidates(tmp_path, monkeypatch):
    from src.rag import graph_extractor

    monkeypatch.setattr(graph_extractor, "chunk_text", lambda text, chunk_size, overlap: [
        "The steering column is mounted to the instrument panel."
    ])
    monkeypatch.setattr(graph_extractor, "route_and_call", lambda **kwargs: {
        "answer": (
            '{"entities":[{"name":"Steering Column","entity_type":"component"},'
            '{"name":"Instrument Panel","entity_type":"component"}],'
            '"relationships":[{"source_name":"Steering Column","source_type":"component",'
            '"target_name":"Instrument Panel","target_type":"component",'
            '"predicate":"mounted_to","evidence_text":"The steering column is mounted to the instrument panel."}]}'
        ),
        "provider": "eva_local", "model": "gemma3:12b",
    })
    result = graph_extractor.extract_graph_candidates("The steering column is mounted to the instrument panel.")
    assert len(result["entities"]) == 2
    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["chunk_id"].startswith("EVA-GRAPH-CHUNK-")
    assert result["model"] == "gemma3:12b"


def test_local_graph_extractor_handles_malformed_json_without_cloud_fallback(monkeypatch):
    from src.rag import graph_extractor

    monkeypatch.setattr(graph_extractor, "chunk_text", lambda text, chunk_size, overlap: ["source excerpt"])
    calls = []
    def local_call(**kwargs):
        calls.append(kwargs)
        return {"answer": "not json", "provider": "eva_local", "model": "local-test"}
    monkeypatch.setattr(graph_extractor, "route_and_call", local_call)
    result = graph_extractor.extract_graph_candidates("source excerpt")
    assert calls and calls[0]["temperature"] == 0.0
    assert result["entities"] == []
    assert result["relationships"] == []
    assert result["warnings"]


def test_graph_extract_endpoint_persists_only_proposals(tmp_path, monkeypatch):
    import asyncio
    from src.rag import rag_routes_v2, graph_extractor

    source = tmp_path / "source.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-a", workspace_id="cad",
        actor_id="user-a", classification="confidential",
    )
    monkeypatch.setattr(rag_routes_v2, "get_record", lambda data_id, **kwargs: {
        **record, "file_path": str(source),
    })
    monkeypatch.setattr(rag_routes_v2, "sha256_file", lambda path: record["sha256"])
    monkeypatch.setattr(graph_extractor, "extract_graph_candidates", lambda text, max_chunks: {
        "entities": [{
            "name": "Steering Column", "entity_type": "component",
            "chunk_id": "chunk-a", "evidence_text": "The steering column is mounted to the instrument panel.",
        }],
        "relationships": [], "chunks_processed": 1, "model": "local-test", "warnings": [],
    })
    response = asyncio.run(rag_routes_v2.rag_graph_extract(
        rag_routes_v2.GraphExtractionRequest(data_id=record["data_id"], workspace_id="cad"),
        user_id="user-a",
    ))
    assert response["status"] == "proposed"
    assert response["approval_required"] is True
    assert response["extraction"]["provider"] == "eva_local"


def test_graph_proposal_review_and_query_flow(tmp_path, monkeypatch):
    import asyncio
    from src.rag import rag_routes_v2

    source = tmp_path / "source.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-a", workspace_id="cad",
        actor_id="user-a", classification="confidential",
    )
    monkeypatch.setattr(rag_routes_v2, "get_record", lambda data_id, **kwargs: {
        **record, "file_path": str(source),
    })
    monkeypatch.setattr(rag_routes_v2, "sha256_file", lambda path: record["sha256"])

    request = rag_routes_v2.GraphProposalRequest(
        data_id=record["data_id"], workspace_id="cad",
        entities=[
            {"name": "Steering Column", "entity_type": "component", "chunk_id": "c1",
             "evidence_text": "The steering column is mounted to the instrument panel."},
            {"name": "Instrument Panel", "entity_type": "component", "chunk_id": "c1",
             "evidence_text": "The steering column is mounted to the instrument panel."},
        ],
        relationships=[{
            "source_name": "Steering Column", "source_type": "component",
            "target_name": "Instrument Panel", "target_type": "component",
            "predicate": "mounted_to", "chunk_id": "c1",
            "evidence_text": "The steering column is mounted to the instrument panel.",
        }],
    )
    proposed = asyncio.run(rag_routes_v2.rag_graph_proposals(request, user_id="user-a"))
    assert proposed["approval_required"] is True
    assert proposed["status"] == "proposed"
    relationship_evidence_id = proposed["relationships"][0]["evidence_id"]

    reviewed = asyncio.run(rag_routes_v2.rag_graph_review(
        rag_routes_v2.GraphEvidenceReviewRequest(
            evidence_id=relationship_evidence_id, status="approved", workspace_id="cad",
        ),
        user_id="user-a",
    ))
    assert reviewed["status"] == "approved"

    result = asyncio.run(rag_routes_v2.rag_graph_query(
        rag_routes_v2.GraphQueryRequest(query="Steering Column", workspace_id="cad"),
        user_id="user-a",
    ))
    assert result["retrieval_mode"] == "approved_graph_only"
    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["data_id"] == record["data_id"]


def test_graph_proposal_batch_validates_all_evidence_before_writing(tmp_path, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2

    source = tmp_path / "source.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-a", workspace_id="cad",
        actor_id="user-a", classification="confidential",
    )
    monkeypatch.setattr(rag_routes_v2, "get_record", lambda data_id, **kwargs: {
        **record, "file_path": str(source),
    })
    monkeypatch.setattr(rag_routes_v2, "sha256_file", lambda path: record["sha256"])
    request = rag_routes_v2.GraphProposalRequest(
        data_id=record["data_id"], workspace_id="cad",
        entities=[
            {"name": "Steering Column", "entity_type": "component", "chunk_id": "c1",
             "evidence_text": "The steering column is mounted"},
            {"name": "Invented Component", "entity_type": "component", "chunk_id": "c2",
             "evidence_text": "This phrase does not occur in the source"},
        ],
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(rag_routes_v2.rag_graph_proposals(request, user_id="user-a"))
    assert error.value.status_code == 422
    assert graph_store.search_entities("Steering Column", owner_id="user-a", workspace_id="cad") == []


def test_graph_proposal_endpoint_requires_source_excerpt_and_authentication(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.rag import rag_routes_v2

    source = tmp_path / "source.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-a", workspace_id="cad",
        actor_id="user-a", classification="confidential",
    )
    monkeypatch.setattr(rag_routes_v2, "get_record", lambda data_id, **kwargs: {
        **record, "file_path": str(source),
    })
    monkeypatch.setattr(rag_routes_v2, "sha256_file", lambda path: record["sha256"])

    async def run_invalid():
        request = rag_routes_v2.GraphProposalRequest(
            data_id=record["data_id"], workspace_id="cad",
            entities=[{
                "name": "Steering Column", "entity_type": "component",
                "chunk_id": "c1", "evidence_text": "This claim is not in the source.",
            }],
        )
        return await rag_routes_v2.rag_graph_proposals(request, user_id="user-a")

    with pytest.raises(HTTPException) as error:
        import asyncio
        asyncio.run(run_invalid())
    assert error.value.status_code == 422


def test_graph_proposal_endpoint_rejects_stale_governed_source(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.rag import rag_routes_v2

    source = tmp_path / "source.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-a", workspace_id="cad",
        actor_id="user-a", classification="confidential",
    )
    monkeypatch.setattr(rag_routes_v2, "get_record", lambda data_id, **kwargs: {
        **record, "file_path": str(source),
    })
    monkeypatch.setattr(rag_routes_v2, "sha256_file", lambda path: "0" * 64)

    async def run_invalid():
        request = rag_routes_v2.GraphProposalRequest(
            data_id=record["data_id"], workspace_id="cad",
            entities=[{
                "name": "Steering Column", "entity_type": "component",
                "chunk_id": "c1", "evidence_text": "The steering column is mounted",
            }],
        )
        return await rag_routes_v2.rag_graph_proposals(request, user_id="user-a")

    with pytest.raises(HTTPException) as error:
        import asyncio
        asyncio.run(run_invalid())
    assert error.value.status_code == 409


def test_url_ingest_url_validation_rejects_private_and_local_destinations(monkeypatch):
    from src.rag.rag_routes_v2 import _validate_public_ingest_url

    with pytest.raises(ValueError):
        _validate_public_ingest_url("http://127.0.0.1/admin")
    with pytest.raises(ValueError):
        _validate_public_ingest_url("http://localhost/")
    with pytest.raises(ValueError):
        _validate_public_ingest_url("http://192.168.1.10/")
    with pytest.raises(ValueError):
        _validate_public_ingest_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        _validate_public_ingest_url("https://example.com:8443/")


def test_url_ingest_url_validation_rejects_dns_with_private_answer(monkeypatch):
    from src.rag import rag_routes_v2

    monkeypatch.setattr(
        rag_routes_v2.socket, "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("10.0.0.5", 443))],
    )
    with pytest.raises(ValueError, match="public IP"):
        rag_routes_v2._validate_public_ingest_url("https://internal.example/")


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
    monkeypatch.setattr(rag, "_get_chroma", lambda: (None, None))
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



def test_approved_reconciliation_repair_reindexes_same_governed_data_id(tmp_path, monkeypatch):
    import asyncio
    from src.rag import rag_routes_v2 as routes

    source = tmp_path / "repair.txt"
    source.write_text("unchanged governed content", encoding="utf-8")
    record = {
        "data_id": "EVA-DATA-REPAIR-1",
        "owner_id": "user-a",
        "workspace_id": "cad",
        "file_path": str(source),
        "file_name": "repair.txt",
        "sha256": "expected-hash",
        "classification": "restricted",
    }
    captured = {}

    monkeypatch.setattr(routes, "get_record", lambda data_id: record)
    monkeypatch.setattr(routes, "record_governance_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        routes,
        "reconcile_data_sources",
        lambda user_id, workspace_id=None: {
            "missing_from_index": [{"data_id": record["data_id"]}]
        },
    )
    monkeypatch.setattr(routes, "sha256_file", lambda path: "expected-hash")
    monkeypatch.setattr(routes, "_extract_text", lambda path, ext: "recovered document text")
    monkeypatch.setattr(routes, "chunk_text", lambda text: ["recovered document text"])
    monkeypatch.setattr(routes.upload_policy, "validate_upload_size", lambda size: None)
    monkeypatch.setattr(routes.upload_policy, "validate_chunk_count", lambda count: None)

    def fake_index_document(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "chunks": 1,
            "vector_indexed": False,
            "vectorless_indexed": True,
            "warnings": [],
        }

    monkeypatch.setattr(routes, "index_document", fake_index_document)

    request = routes.ReconciliationRepairRequest(
        data_id=record["data_id"],
        action="reindex_missing",
        approved=True,
        workspace_id="cad",
    )
    result = asyncio.run(routes.rag_reconciliation_repair(request, user_id="user-a"))

    assert result["ok"] is True
    assert result["data_id"] == record["data_id"]
    assert result["source_file_deleted"] is False
    assert result["governance_record_changed"] is False
    assert captured["metadata"]["data_id"] == record["data_id"]
    assert captured["user_id"] == "user-a"
    assert captured["workspace_id"] == "cad"


def test_reconciliation_repair_requires_explicit_approval():
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2 as routes

    request = routes.ReconciliationRepairRequest(
        data_id="EVA-DATA-REPAIR-2",
        action="reindex_missing",
        approved=False,
        workspace_id="cad",
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_reconciliation_repair(request, user_id="user-a"))

    assert error.value.status_code == 400
    assert "approval" in error.value.detail.lower()



def test_reconciliation_repair_blocks_modified_source_content(tmp_path, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2 as routes

    source = tmp_path / "modified.txt"
    source.write_text("changed since governance approval", encoding="utf-8")
    record = {
        "data_id": "EVA-DATA-REPAIR-3",
        "owner_id": "user-a",
        "workspace_id": "cad",
        "file_path": str(source),
        "file_name": "modified.txt",
        "sha256": "old-hash",
        "classification": "restricted",
    }
    monkeypatch.setattr(routes, "get_record", lambda data_id: record)
    monkeypatch.setattr(
        routes,
        "reconcile_data_sources",
        lambda user_id, workspace_id=None: {
            "missing_from_index": [{"data_id": record["data_id"]}]
        },
    )
    monkeypatch.setattr(routes, "sha256_file", lambda path: "new-hash")
    monkeypatch.setattr(
        routes,
        "index_document",
        lambda **kwargs: pytest.fail("Modified content must not be re-indexed automatically"),
    )

    request = routes.ReconciliationRepairRequest(
        data_id=record["data_id"],
        action="reindex_missing",
        approved=True,
        workspace_id="cad",
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_reconciliation_repair(request, user_id="user-a"))

    assert error.value.status_code == 409
    assert "revision review" in error.value.detail.lower()



def test_orphan_cleanup_deletes_only_exact_owner_workspace_data_id(monkeypatch):
    from src.rag import hybrid_rag_v2 as rag

    class FakeCollection:
        def __init__(self):
            self.deleted = []

        def count(self):
            return 3

        def get(self, where=None, limit=None, offset=0, include=None):
            rows = [
                ("chunk-a", {"user_id": "user-a", "workspace_id": "cad", "data_id": "ORPHAN-A"}),
                ("chunk-b", {"user_id": "user-a", "workspace_id": "cad", "data_id": "VALID-B"}),
                ("chunk-c", {"user_id": "user-b", "workspace_id": "cad", "data_id": "ORPHAN-A"}),
            ][offset:offset + limit]
            return {"ids": [row[0] for row in rows], "metadatas": [row[1] for row in rows]}

        def delete(self, ids):
            self.deleted.extend(ids)

    collection = FakeCollection()
    monkeypatch.setattr(rag, "_get_chroma", lambda: (object(), collection))
    monkeypatch.setattr(rag, "_vectorless_chunks", [
        {
            "id": "vl-a",
            "user_id": "user-a",
            "workspace_id": "cad",
            "metadata": {"user_id": "user-a", "workspace_id": "cad", "data_id": "ORPHAN-A"},
        },
        {
            "id": "vl-b",
            "user_id": "user-a",
            "workspace_id": "cad",
            "metadata": {"user_id": "user-a", "workspace_id": "other", "data_id": "ORPHAN-A"},
        },
        {
            "id": "vl-c",
            "user_id": "user-b",
            "workspace_id": "cad",
            "metadata": {"user_id": "user-b", "workspace_id": "cad", "data_id": "ORPHAN-A"},
        },
    ])

    result = rag.delete_orphaned_data_id(
        "ORPHAN-A", user_id="user-a", workspace_id="cad"
    )

    assert collection.deleted == ["chunk-a"]
    assert result["deleted_vector_chunks"] == 1
    assert result["deleted_vectorless_chunks"] == 1
    assert [item["id"] for item in rag._vectorless_chunks] == ["vl-b", "vl-c"]


def test_orphan_cleanup_requires_current_orphan_and_explicit_approval(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2 as routes

    request = routes.OrphanIndexCleanupRequest(
        data_id="ORPHAN-A", approved=False, workspace_id="cad"
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_reconciliation_cleanup_orphan(request, user_id="user-a"))
    assert error.value.status_code == 400

    request.approved = True
    monkeypatch.setattr(
        routes, "reconcile_data_sources",
        lambda user_id, workspace_id=None: {"index_without_governance": []},
    )
    monkeypatch.setattr(
        routes, "delete_orphaned_data_id",
        lambda *args, **kwargs: pytest.fail("Non-orphan data must never be deleted"),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_reconciliation_cleanup_orphan(request, user_id="user-a"))
    assert error.value.status_code == 409


def test_orphan_cleanup_endpoint_records_audit_and_never_deletes_source(monkeypatch):
    import asyncio
    from src.rag import rag_routes_v2 as routes

    captured = {}
    request = routes.OrphanIndexCleanupRequest(
        data_id="ORPHAN-A", approved=True, workspace_id="cad"
    )
    monkeypatch.setattr(
        routes, "reconcile_data_sources",
        lambda user_id, workspace_id=None: {
            "index_without_governance": [{"data_id": "ORPHAN-A"}]
        },
    )
    monkeypatch.setattr(
        routes, "delete_orphaned_data_id",
        lambda *args, **kwargs: {
            "ok": True, "data_id": "ORPHAN-A",
            "deleted_vector_chunks": 2,
            "deleted_vectorless_chunks": 1,
            "deleted_chunks": 3,
        },
    )
    monkeypatch.setattr(
        routes, "record_reconciliation_event",
        lambda *args, **kwargs: captured.update({"audit": (args, kwargs)}),
    )

    result = asyncio.run(routes.rag_reconciliation_cleanup_orphan(request, user_id="user-a"))

    assert result["deleted_chunks"] == 3
    assert result["source_files_deleted"] is False
    assert result["governance_records_changed"] is False
    assert result["audit_recorded"] is True
    assert captured["audit"][1]["action"] == "orphan_rag_index_cleaned"


def test_rag_owner_filter_does_not_treat_anonymous_or_default_as_wildcard():
    from src.rag import hybrid_rag_v2 as rag

    private_chunk = {
        "user_id": "user-a",
        "workspace_id": "cad",
        "metadata": {"user_id": "user-a", "workspace_id": "cad"},
    }
    assert rag._chunk_user_matches(private_chunk, "user-a", "cad") is True
    assert rag._chunk_user_matches(private_chunk, "user-b", "cad") is False
    assert rag._chunk_user_matches(private_chunk, "anonymous", "cad") is False
    assert rag._chunk_user_matches(private_chunk, "default", "cad") is False


def test_chroma_bm25_fetch_always_filters_by_exact_owner(monkeypatch):
    from src.rag import hybrid_rag_v2 as rag

    class FakeCollection:
        def count(self):
            return 1

        def get(self, **kwargs):
            self.where = kwargs.get("where")
            return {
                "ids": ["private-1"],
                "documents": ["private engineering data"],
                "metadatas": [{"user_id": "user-a", "workspace_id": "cad"}],
            }

    collection = FakeCollection()
    monkeypatch.setattr(rag, "_get_chroma", lambda: (object(), collection))

    assert rag._chroma_chunks_for_user("anonymous") == []
    assert collection.where == {"user_id": "anonymous"}
    assert rag._chroma_chunks_for_user("user-a", workspace_id="cad")[0]["text"] == "private engineering data"
    assert collection.where == {"user_id": "user-a"}


def test_rag_query_rejects_anonymous_user():
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2 as routes

    request = routes.QueryRequest(query="private document question")
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_query(request, user_id="anonymous"))
    assert error.value.status_code == 401


def test_rag_chat_rejects_anonymous_user():
    import asyncio
    from fastapi import HTTPException
    from src.rag import rag_routes_v2 as routes

    request = routes.ChatRequest(message="private document question")
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.rag_chat(request, user_id="anonymous"))
    assert error.value.status_code == 401

def test_jwt_without_user_identity_fails_closed(monkeypatch):
    from fastapi.security import HTTPAuthorizationCredentials
    from src.rag import rag_routes_v2 as routes
    import jose

    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setattr(jose.jwt, "decode", lambda *args, **kwargs: {})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="signed-token")
    assert routes.get_user_id(credentials) == "anonymous"


def test_delete_indexed_document_is_owner_and_workspace_scoped(monkeypatch):
    from src.rag import hybrid_rag_v2 as rag

    class FakeCollection:
        def __init__(self):
            self.deleted = []
            self.where_values = []

        def count(self):
            return 2

        def get(self, **kwargs):
            self.where_values.append(kwargs.get("where"))
            return {
                "ids": ["user-a-doc", "user-b-doc"],
                "metadatas": [
                    {"user_id": "user-a", "workspace_id": "cad", "source": "shared.txt"},
                    {"user_id": "user-b", "workspace_id": "cad", "source": "shared.txt"},
                ],
            }

        def delete(self, ids):
            self.deleted.extend(ids)

    collection = FakeCollection()
    monkeypatch.setattr(rag, "_get_chroma", lambda: (object(), collection))
    result = rag.delete_indexed_document("shared.txt", "user-a", workspace_id="cad")

    assert result["deleted_chunks"] == 1
    assert collection.deleted == ["user-a-doc"]
    assert collection.where_values == [{"user_id": "user-a"}]
