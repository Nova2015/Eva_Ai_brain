import hashlib
from pathlib import Path

import pytest

from src.data_governance import (
    get_record,
    list_versions,
    record_revision,
    register_import,
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
