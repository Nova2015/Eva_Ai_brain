import hashlib
from pathlib import Path

import pytest

from src.data_governance import (
    get_record,
    list_records,
    list_versions,
    record_revision,
    register_import,
    recover_location,
    set_training_approval,
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
    assert get_record(first["data_id"])["data_id"] == first["data_id"]


def test_rename_and_move_preserves_data_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    original = tmp_path / "source.txt"
    original.write_text("engineering data", encoding="utf-8")
    record = _register(original)

    moved = tmp_path / "renamed.txt"
    original.rename(moved)
    updated = update_location(record["data_id"], moved, actor_id="user-1")

    assert updated["data_id"] == record["data_id"]
    assert updated["file_name"] == "renamed.txt"
    assert updated["sha256"] == hashlib.sha256(b"engineering data").hexdigest()


def test_content_change_creates_new_version(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "part.step"
    source.write_text("revision one", encoding="utf-8")
    record = _register(source)

    unchanged = record_revision(record["data_id"], actor_id="user-1")
    assert unchanged["changed"] is False
    assert unchanged["version"] == 1

    source.write_text("revision two", encoding="utf-8")
    changed = record_revision(record["data_id"], actor_id="user-1")
    assert changed["changed"] is True
    assert changed["version"] == 2
    assert [v["version"] for v in list_versions(record["data_id"])] == [1, 2]


def test_training_data_requires_explicit_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / "sample.jsonl"
    source.write_text('{"text":"example"}', encoding="utf-8")
    record = _register(source)

    assert get_record(record["data_id"])["training_approval"] == "pending"
    approved = set_training_approval(record["data_id"], "approved", actor_id="reviewer-1")
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
    assert list_audit_events(record["data_id"])[-1]["action"] == "location_recovered"


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
