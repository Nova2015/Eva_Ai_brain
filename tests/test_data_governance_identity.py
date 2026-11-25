import hashlib
import sqlite3
from pathlib import Path

import pytest

from src.data_governance import (
    get_record,
    authorize_data_action,
    approve_content_revision,
    list_records,
    list_reconciliation_events,
    list_audit_events,
    list_versions,
    record_revision,
    register_import,
    recover_location,
    record_governance_event,
    record_reconciliation_event,
    set_training_approval,
    set_classification,
    set_permissions,
    update_location,
)


def _register(path: Path):
    return register_import(
        path,
        source="local_upload",
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
    )


def test_each_import_gets_distinct_persistent_data_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("same bytes", encoding="utf-8")
    second_file.write_text("same bytes", encoding="utf-8")

    first = _register(first_file)
    second = _register(second_file)

    assert first["data_id"] != second["data_id"]
    assert first["sha256"] == second["sha256"]
    assert first["training_approval"] == "pending"
    assert get_record(first["data_id"], owner_id="user-1", workspace_id="workspace-1")["data_id"] == first["data_id"]


def test_rename_and_move_preserves_data_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    original = tmp_path / "source.txt"
    original.write_text("engineering data", encoding="utf-8")
    record = _register(original)

    moved = tmp_path / "renamed.txt"
    original.rename(moved)
    updated = update_location(record["data_id"], moved, owner_id="user-1", workspace_id="workspace-1", actor_id="user-1")

    assert updated["data_id"] == record["data_id"]
    assert updated["file_name"] == "renamed.txt"
    assert updated["sha256"] == hashlib.sha256(b"engineering data").hexdigest()


def test_content_change_creates_new_version(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "part.step"
    source.write_text("revision one", encoding="utf-8")
    record = _register(source)

    unchanged = record_revision(record["data_id"], owner_id="user-1", workspace_id="workspace-1", actor_id="user-1")
    assert unchanged["changed"] is False
    assert unchanged["version"] == 1

    source.write_text("revision two", encoding="utf-8")
    changed = record_revision(record["data_id"], owner_id="user-1", workspace_id="workspace-1", actor_id="user-1")
    assert changed["changed"] is True
    assert changed["version"] == 2
    assert [v["version"] for v in list_versions(record["data_id"], owner_id="user-1", workspace_id="workspace-1")] == [1, 2]


def test_training_data_requires_explicit_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "sample.jsonl"
    source.write_text('{"text":"example"}', encoding="utf-8")
    record = _register(source)

    assert get_record(record["data_id"], owner_id="user-1", workspace_id="workspace-1")["training_approval"] == "pending"
    approved = set_training_approval(record["data_id"], "approved", owner_id="user-1", workspace_id="workspace-1", actor_id="reviewer-1")
    assert approved["training_approval"] == "approved"


def test_invalid_classification_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "sample.txt"
    source.write_text("private", encoding="utf-8")

    with pytest.raises(ValueError):
        register_import(
            source,
            source="local_upload",
            owner_id="user-1",
            workspace_id="workspace-1",
            actor_id="user-1",
            classification="unknown",
        )



def test_list_records_is_scoped_to_owner_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    third_file = tmp_path / "third.txt"
    first_file.write_text("owner one workspace one", encoding="utf-8")
    second_file.write_text("owner one workspace two", encoding="utf-8")
    third_file.write_text("owner two workspace one", encoding="utf-8")

    first = register_import(
        first_file, source="test", owner_id="user-1", workspace_id="workspace-1",
        actor_id="user-1",
    )
    register_import(
        second_file, source="test", owner_id="user-1", workspace_id="workspace-2",
        actor_id="user-1",
    )
    register_import(
        third_file, source="test", owner_id="user-2", workspace_id="workspace-1",
        actor_id="user-2",
    )

    scoped = list_records(owner_id="user-1", workspace_id="workspace-1")
    owner_records = list_records(owner_id="user-1")

    assert [record["data_id"] for record in scoped] == [first["data_id"]]
    assert len(owner_records) == 2
    assert all(record["owner_id"] == "user-1" for record in owner_records)



def test_recover_location_preserves_data_id_and_requires_matching_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    governed_root = tmp_path / "governed"
    governed_root.mkdir()
    original = governed_root / "original.txt"
    recovered = governed_root / "renamed.txt"
    original.write_text("same approved bytes", encoding="utf-8")
    record = register_import(
        original,
        source="test",
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
    )
    original.rename(recovered)

    updated = recover_location(
        record["data_id"],
        recovered,
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
        allowed_root=governed_root,
    )

    assert updated["data_id"] == record["data_id"]
    assert updated["file_path"] == str(recovered.resolve())
    assert updated["version"] == record["version"]
    assert list_audit_events(record["data_id"], owner_id="user-1", workspace_id="workspace-1")[-1]["action"] == "location_recovered"


def test_recover_location_rejects_hash_mismatch_and_outside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    governed_root = tmp_path / "governed"
    governed_root.mkdir()
    original = governed_root / "original.txt"
    changed = governed_root / "changed.txt"
    outside = tmp_path / "outside.txt"
    original.write_text("approved bytes", encoding="utf-8")
    changed.write_text("different bytes", encoding="utf-8")
    outside.write_text("approved bytes", encoding="utf-8")
    record = register_import(
        original,
        source="test",
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        recover_location(
            record["data_id"], changed, owner_id="user-1",
            workspace_id="workspace-1", actor_id="user-1",
            allowed_root=governed_root,
        )
    with pytest.raises(ValueError, match="inside"):
        recover_location(
            record["data_id"], outside, owner_id="user-1",
            workspace_id="workspace-1", actor_id="user-1",
            allowed_root=governed_root,
        )



def test_approved_content_revision_increments_version_and_audits(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "revision.txt"
    source.write_text("version one", encoding="utf-8")
    record = _register(source)
    source.write_text("version two", encoding="utf-8")

    result = approve_content_revision(
        record["data_id"],
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
        expected_sha256=record["sha256"],
    )

    assert result["changed"] is True
    assert result["version"] == 2
    assert list_versions(record["data_id"], owner_id="user-1", workspace_id="workspace-1")[-1]["sha256"] == result["sha256"]
    assert list_audit_events(record["data_id"], owner_id="user-1", workspace_id="workspace-1")[-1]["action"] == "content_revision_approved"


def test_content_revision_approval_rejects_stale_hash_and_wrong_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "revision.txt"
    source.write_text("version one", encoding="utf-8")
    record = _register(source)
    source.write_text("version two", encoding="utf-8")

    with pytest.raises(KeyError):
        approve_content_revision(
            record["data_id"], owner_id="other-user",
            workspace_id="workspace-1", actor_id="other-user",
            expected_sha256=record["sha256"],
        )
    with pytest.raises(ValueError, match="revision changed"):
        approve_content_revision(
            record["data_id"], owner_id="user-1",
            workspace_id="workspace-1", actor_id="user-1",
            expected_sha256="stale-hash",
        )



def test_governance_events_are_owner_scoped_and_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "audit.txt"
    source.write_text("audit data", encoding="utf-8")
    record = _register(source)

    with pytest.raises(KeyError):
        record_governance_event(
            record["data_id"], "unauthorized_action",
            owner_id="other-user", workspace_id="workspace-1",
            actor_id="other-user",
        )

    record_governance_event(
        record["data_id"], "approved_repair",
        owner_id="user-1", workspace_id="workspace-1",
        actor_id="user-1", details={"action": "reindex_missing"},
    )
    events = list_audit_events(record["data_id"], owner_id="user-1", workspace_id="workspace-1")
    assert events[-1]["action"] == "approved_repair"
    assert events[-1]["actor_id"] == "user-1"



def test_reconciliation_audit_supports_orphan_without_governance_record(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))

    record_reconciliation_event(
        "ORPHAN-DATA-ID",
        owner_id="user-1",
        workspace_id="workspace-1",
        action="orphan_rag_index_cleaned",
        actor_id="user-1",
        details={"deleted_chunks": 4, "source_files_deleted": False},
    )

    events = list_reconciliation_events(
        owner_id="user-1",
        workspace_id="workspace-1",
        data_id="ORPHAN-DATA-ID",
    )
    assert len(events) == 1
    assert events[0]["action"] == "orphan_rag_index_cleaned"
    assert events[0]["details"]["source_files_deleted"] is False
    assert list_reconciliation_events(
        owner_id="other-user", workspace_id="workspace-1",
        data_id="ORPHAN-DATA-ID",
    ) == []



def test_data_item_reads_and_mutations_reject_wrong_owner_or_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "scoped.txt"
    source.write_text("owner-scoped data", encoding="utf-8")
    record = _register(source)
    data_id = record["data_id"]

    wrong_scope = {"owner_id": "other-user", "workspace_id": "workspace-1"}
    assert get_record(data_id, **wrong_scope) is None
    with pytest.raises(KeyError):
        update_location(
            data_id, source, **wrong_scope, actor_id="other-user"
        )
    with pytest.raises(KeyError):
        record_revision(
            data_id, **wrong_scope, actor_id="other-user"
        )
    with pytest.raises(KeyError):
        set_classification(
            data_id, "restricted", **wrong_scope, actor_id="other-user"
        )
    with pytest.raises(KeyError):
        set_permissions(
            data_id, {"read": ["other-user"]}, **wrong_scope,
            actor_id="other-user",
        )
    with pytest.raises(KeyError):
        set_training_approval(
            data_id, "approved", **wrong_scope, actor_id="other-user"
        )
    with pytest.raises(KeyError):
        list_versions(data_id, **wrong_scope)
    with pytest.raises(KeyError):
        list_audit_events(data_id, **wrong_scope)

    # A different workspace under the same owner is also isolated.
    wrong_workspace = {"owner_id": "user-1", "workspace_id": "other-workspace"}
    assert get_record(data_id, **wrong_workspace) is None
    with pytest.raises(KeyError):
        list_audit_events(data_id, **wrong_workspace)


def test_permission_matrix_has_independent_read_write_execute_delete_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "permissions.txt"
    source.write_text("permission defaults", encoding="utf-8")
    record = _register(source)

    assert record["permissions"] == {
        "read": ["user-1"],
        "write": ["user-1"],
        "execute": [],
        "delete": [],
    }

    updated = set_permissions(
        record["data_id"],
        {
            "read": ["user-2", "user-1", "user-2"],
            "write": ["user-1"],
            "execute": [],
            "delete": ["admin-1"],
        },
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
    )
    assert updated["permissions"] == {
        "read": ["user-1", "user-2"],
        "write": ["user-1"],
        "execute": [],
        "delete": ["admin-1"],
    }


def test_permission_matrix_rejects_unknown_actions_and_invalid_principals(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "permissions.txt"
    source.write_text("permission validation", encoding="utf-8")
    record = _register(source)

    with pytest.raises(ValueError, match="Unsupported permission actions"):
        set_permissions(
            record["data_id"], {"read": ["user-1"], "admin": ["user-1"]},
            owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
        )
    with pytest.raises(ValueError, match="invalid principal"):
        set_permissions(
            record["data_id"], {"read": [""]},
            owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
        )


def test_acl_grants_only_the_configured_independent_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "shared.txt"
    source.write_text("shared engineering data", encoding="utf-8")
    record = _register(source)

    set_permissions(
        record["data_id"],
        {
            "read": ["user-2"],
            "write": ["user-2"],
            "execute": [],
            "delete": [],
        },
        owner_id="user-1",
        workspace_id="workspace-1",
        actor_id="user-1",
    )

    shared = get_record(
        record["data_id"], owner_id="user-2", workspace_id="workspace-1"
    )
    assert shared is not None
    assert shared["owner_id"] == "user-1"
    assert authorize_data_action(
        record["data_id"], actor_id="user-2",
        workspace_id="workspace-1", action="read",
    )
    assert authorize_data_action(
        record["data_id"], actor_id="user-2",
        workspace_id="workspace-1", action="write",
    )
    assert not authorize_data_action(
        record["data_id"], actor_id="user-2",
        workspace_id="workspace-1", action="execute",
    )
    assert not authorize_data_action(
        record["data_id"], actor_id="user-2",
        workspace_id="workspace-1", action="delete",
    )

    updated = set_classification(
        record["data_id"], "confidential",
        owner_id="user-2", workspace_id="workspace-1", actor_id="user-2",
    )
    assert updated["classification"] == "confidential"
    assert get_record(
        record["data_id"], owner_id="user-3", workspace_id="workspace-1"
    ) is None


def test_acl_does_not_grant_access_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "workspace.txt"
    source.write_text("workspace-bound data", encoding="utf-8")
    record = _register(source)
    set_permissions(
        record["data_id"],
        {"read": ["user-2"], "write": [], "execute": [], "delete": []},
        owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
    )

    assert get_record(
        record["data_id"], owner_id="user-2", workspace_id="workspace-2"
    ) is None
    assert not authorize_data_action(
        record["data_id"], actor_id="user-2",
        workspace_id="workspace-2", action="read",
    )


def test_write_acl_allows_scoped_revision_and_location_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    governed_root = tmp_path / "governed"
    governed_root.mkdir()
    source = governed_root / "original.txt"
    source.write_text("revision one", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-1", workspace_id="workspace-1",
        actor_id="user-1",
    )
    set_permissions(
        record["data_id"],
        {
            "read": ["user-1", "user-2"],
            "write": ["user-1", "user-2"],
            "execute": [],
            "delete": [],
        },
        owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
    )

    source.write_text("revision two", encoding="utf-8")
    approved = approve_content_revision(
        record["data_id"],
        owner_id="user-2",
        workspace_id="workspace-1",
        actor_id="user-2",
        expected_sha256=record["sha256"],
    )
    assert approved["changed"] is True
    assert approved["version"] == 2

    recovered_path = governed_root / "recovered.txt"
    source.rename(recovered_path)
    recovered = recover_location(
        record["data_id"],
        recovered_path,
        owner_id="user-2",
        workspace_id="workspace-1",
        actor_id="user-2",
        allowed_root=governed_root,
    )
    assert recovered["file_path"] == str(recovered_path.resolve())
    assert recovered["version"] == 2
    actions = [
        event["action"]
        for event in list_audit_events(
            record["data_id"], owner_id="user-2", workspace_id="workspace-1"
        )
    ]
    assert "content_revision_approved" in actions
    assert "location_recovered" in actions


def test_update_location_rejects_replacement_content_with_different_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    original = tmp_path / "original.txt"
    replacement = tmp_path / "replacement.txt"
    original.write_text("approved bytes", encoding="utf-8")
    replacement.write_text("different bytes", encoding="utf-8")
    record = _register(original)

    with pytest.raises(ValueError, match="SHA-256"):
        update_location(
            record["data_id"], replacement,
            owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
        )

    current = get_record(
        record["data_id"], owner_id="user-1", workspace_id="workspace-1"
    )
    assert current["file_path"] == str(original.resolve())
    assert current["sha256"] == record["sha256"]


def test_audited_mutations_require_actor_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "actor.txt"
    source.write_text("actor attribution", encoding="utf-8")
    record = _register(source)
    scope = {"owner_id": "user-1", "workspace_id": "workspace-1"}

    with pytest.raises(ValueError, match="actor_id"):
        update_location(record["data_id"], source, **scope, actor_id="")
    with pytest.raises(ValueError, match="actor_id"):
        record_revision(record["data_id"], **scope, actor_id="")
    with pytest.raises(ValueError, match="actor_id"):
        set_classification(
            record["data_id"], "confidential", **scope, actor_id=""
        )
    with pytest.raises(ValueError, match="actor_id"):
        set_permissions(
            record["data_id"], {"read": ["user-1"]}, **scope, actor_id=""
        )
    with pytest.raises(ValueError, match="actor_id"):
        set_training_approval(
            record["data_id"], "rejected", **scope, actor_id=""
        )


def test_audit_tables_are_append_only_even_for_direct_sql(tmp_path, monkeypatch):
    db_path = tmp_path / "governance.db"
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(db_path))
    source = tmp_path / "audit-immutable.txt"
    source.write_text("immutable audit source", encoding="utf-8")
    record = _register(source)
    record_governance_event(
        record["data_id"], "test_event",
        owner_id="user-1", workspace_id="workspace-1", actor_id="user-1",
    )
    record_reconciliation_event(
        "ORPHAN-A", owner_id="user-1", workspace_id="workspace-1",
        action="orphan_seen", actor_id="user-1",
    )

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="data_id is immutable"):
            conn.execute(
                "UPDATE data_items SET data_id='EVA-DATA-TAMPERED' WHERE data_id=?",
                (record["data_id"],),
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE data_item_audit SET action='tampered' WHERE data_id=?",
                (record["data_id"],),
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "DELETE FROM data_item_audit WHERE data_id=?",
                (record["data_id"],),
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE reconciliation_audit SET action='tampered' WHERE data_id='ORPHAN-A'"
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "DELETE FROM reconciliation_audit WHERE data_id='ORPHAN-A'"
            )
        conn.rollback()
    finally:
        conn.close()


@pytest.mark.parametrize("classification", ["public", "private", "sensitive", "ip"])
def test_canonical_governance_classifications_are_supported(tmp_path, monkeypatch, classification):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / f"{classification}.txt"
    source.write_text("classified engineering content", encoding="utf-8")
    record = register_import(
        source, source="test", owner_id="user-1", workspace_id="workspace-1",
        actor_id="user-1", classification=classification,
    )
    assert record["classification"] == classification
