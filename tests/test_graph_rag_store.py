from pathlib import Path

import pytest

from src.data_governance import approve_content_revision, register_import
from src.rag import graph_store


def _governed_source(tmp_path: Path, owner="user-a", workspace="cad"):
    source = tmp_path / f"{owner}-{workspace}.txt"
    source.write_text("The steering column is mounted to the instrument panel.", encoding="utf-8")
    return register_import(
        source,
        source="test",
        owner_id=owner,
        workspace_id=workspace,
        actor_id=owner,
        classification="confidential",
    )


@pytest.fixture(autouse=True)
def isolated_graph_and_governance(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_GRAPH_RAG_DB", str(tmp_path / "graph.db"))


def test_graph_entity_requires_governed_source_and_starts_proposed(tmp_path):
    record = _governed_source(tmp_path)
    entity = graph_store.upsert_entity(
        "Steering Column", "component",
        data_id=record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", chunk_id="chunk-1",
        evidence_text="The steering column is mounted to the instrument panel.",
        confidence=0.91,
    )
    assert entity["status"] == "proposed"
    assert entity["entity_id"].startswith("EVA-ENT-")
    assert entity["evidence_id"].startswith("EVA-GE-")


def test_graph_entities_are_deduplicated_with_separate_evidence(tmp_path):
    record = _governed_source(tmp_path)
    kwargs = dict(
        data_id=record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", evidence_text="Steering column component evidence.",
    )
    first = graph_store.upsert_entity("Steering Column", "component", chunk_id="chunk-1", **kwargs)
    second = graph_store.upsert_entity(" steering   column ", "component", chunk_id="chunk-2", **kwargs)
    assert first["entity_id"] == second["entity_id"]
    assert first["evidence_id"] != second["evidence_id"]


def test_graph_relationship_is_not_retrievable_until_evidence_approved(tmp_path):
    record = _governed_source(tmp_path)
    base = dict(
        data_id=record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", evidence_text="Steering column is mounted to instrument panel.",
    )
    source = graph_store.upsert_entity("Steering Column", "component", chunk_id="c1", **base)
    target = graph_store.upsert_entity("Instrument Panel", "component", chunk_id="c2", **base)
    edge = graph_store.upsert_relationship(
        source["entity_id"], target["entity_id"], "mounted_to",
        chunk_id="c3", **base,
    )
    assert graph_store.traverse_verified_relationships(
        [source["entity_id"]], owner_id="user-a", workspace_id="cad"
    ) == []
    graph_store.review_evidence(
        edge["evidence_id"], "approved",
        owner_id="user-a", workspace_id="cad", actor_id="user-a",
    )
    results = graph_store.traverse_verified_relationships(
        [source["entity_id"]], owner_id="user-a", workspace_id="cad"
    )
    assert len(results) == 1
    assert results[0]["predicate"] == "mounted_to"
    assert results[0]["data_id"] == record["data_id"]
    assert results[0]["classification"] == "confidential"


def test_graph_store_enforces_owner_and_workspace_isolation(tmp_path):
    record = _governed_source(tmp_path, owner="user-a", workspace="cad")
    with pytest.raises(PermissionError):
        graph_store.upsert_entity(
            "Steering Column", "component",
            data_id=record["data_id"], owner_id="user-b", workspace_id="cad",
            actor_id="user-b", chunk_id="chunk-1",
            evidence_text="private evidence",
        )


def test_graph_relationship_rejects_cross_workspace_endpoints(tmp_path):
    record_a = _governed_source(tmp_path, owner="user-a", workspace="cad")
    record_b = _governed_source(tmp_path, owner="user-a", workspace="powertrain")
    a = graph_store.upsert_entity(
        "Steering Column", "component",
        data_id=record_a["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", chunk_id="c1", evidence_text="steering column",
    )
    b = graph_store.upsert_entity(
        "Engine", "component",
        data_id=record_b["data_id"], owner_id="user-a", workspace_id="powertrain",
        actor_id="user-a", chunk_id="c2", evidence_text="engine",
    )
    with pytest.raises(PermissionError):
        graph_store.upsert_relationship(
            a["entity_id"], b["entity_id"], "connected_to",
            data_id=record_a["data_id"], owner_id="user-a", workspace_id="cad",
            actor_id="user-a", chunk_id="c3", evidence_text="connected",
        )


def test_graph_entity_search_returns_only_approved_evidence(tmp_path):
    record = _governed_source(tmp_path)
    entity = graph_store.upsert_entity(
        "Instrument Panel", "component",
        data_id=record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", chunk_id="chunk-1", evidence_text="instrument panel",
    )
    assert graph_store.search_entities("instrument", owner_id="user-a", workspace_id="cad") == []
    graph_store.review_evidence(
        entity["evidence_id"], "approved",
        owner_id="user-a", workspace_id="cad", actor_id="user-a",
    )
    assert len(graph_store.search_entities("instrument", owner_id="user-a", workspace_id="cad")) == 1

def test_graph_evidence_from_old_source_revision_cannot_be_approved(tmp_path):
    record = _governed_source(tmp_path)
    source = graph_store.upsert_entity(
        "Steering Column", "component",
        data_id=record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", chunk_id="chunk-1",
        evidence_text="The steering column is mounted to the instrument panel.",
    )
    source_file = tmp_path / "user-a-cad.txt"
    source_file.write_text("Updated engineering source revision.", encoding="utf-8")
    approve_content_revision(
        record["data_id"], owner_id="user-a", workspace_id="cad",
        actor_id="user-a", expected_sha256=record["sha256"],
    )
    with pytest.raises(ValueError, match="newer revision"):
        graph_store.review_evidence(
            source["evidence_id"], "approved",
            owner_id="user-a", workspace_id="cad", actor_id="user-a",
        )
