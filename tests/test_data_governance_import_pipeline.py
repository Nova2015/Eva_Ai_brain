from pathlib import Path
import json
import sqlite3
import pytest

from src.data_governance.import_pipeline import (
    list_quarantined,
    quarantine_candidate,
    release_quarantined,
    validate_and_register_import,
    validate_import_candidate,
)


def test_valid_text_import_is_registered_with_hash_and_scan_status(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "valid.txt"
    source.write_text("Valid local engineering document", encoding="utf-8")

    result = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", classification="private",
        malware_scanner=lambda path: True,
    )

    assert result["ok"] is True
    assert result["status"] == "registered"
    assert result["validation"]["malware_scan"] == "clean"
    assert result["record"]["sha256"] == result["validation"]["sha256"]
    assert source.exists()


def test_invalid_pdf_is_quarantined_without_modifying_original(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "invalid.pdf"
    original_bytes = b"this is not a PDF file"
    source.write_bytes(original_bytes)

    result = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1",
    )

    assert result["ok"] is False
    assert result["status"] == "quarantined"
    assert "PDF signature is invalid" in result["validation"]["issues"]
    quarantined_path = Path(result["quarantine"]["quarantine_path"])
    manifest_path = quarantined_path.with_suffix(".json")
    assert quarantined_path.read_bytes() == original_bytes
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["original_file_deleted"] is False
    assert source.read_bytes() == original_bytes
    items = list_quarantined(owner_id="owner-1", workspace_id="cad")
    assert len(items) == 1
    assert items[0]["quarantine_id"] == result["quarantine"]["quarantine_id"]
    assert list_quarantined(owner_id="owner-1", workspace_id="other-workspace") == []
    conn = sqlite3.connect(tmp_path / "governance.db")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE quarantine_audit SET action='tampered' WHERE quarantine_id=?",
                (result["quarantine"]["quarantine_id"],),
            )
    finally:
        conn.close()


def test_size_limit_and_unsupported_extension_fail_closed_to_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "large.unknown"
    source.write_text("123456789", encoding="utf-8")

    result = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", max_bytes=4,
    )
    assert result["status"] == "quarantined"
    assert any("configured import limit" in issue for issue in result["validation"]["issues"])
    assert any("Unsupported file extension" in issue for issue in result["validation"]["issues"])
    assert source.read_text(encoding="utf-8") == "123456789"


def test_malware_scanner_rejection_is_quarantined(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "suspicious.txt"
    source.write_text("suspicious content", encoding="utf-8")

    result = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", malware_scanner=lambda path: False,
    )
    assert result["status"] == "quarantined"
    assert result["validation"]["malware_scan"] == "infected"
    assert source.exists()


def test_malware_scanner_errors_fail_closed_to_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "scan-error.txt"
    source.write_text("scan failure content", encoding="utf-8")

    def broken_scanner(path):
        raise RuntimeError("scanner unavailable")

    result = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", malware_scanner=broken_scanner,
    )
    assert result["status"] == "quarantined"
    assert result["validation"]["malware_scan"] == "scan_error"
    assert source.exists()


def test_quarantine_copy_is_sha256_verified(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "candidate.txt"
    source.write_text("quarantine integrity", encoding="utf-8")
    result = quarantine_candidate(
        source, owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reason="manual validation hold",
    )
    assert result["sha256"]
    assert Path(result["quarantine_path"]).read_bytes() == source.read_bytes()
    assert source.exists()


def test_format_validation_rejects_corrupt_json_and_accepts_valid_json(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not valid json", encoding="utf-8")
    rejected = validate_import_candidate(invalid)
    assert rejected["ok"] is False
    assert "JSON content is invalid" in rejected["issues"]

    valid = tmp_path / "valid.json"
    valid.write_text('{"part": "door-trim"}', encoding="utf-8")
    accepted = validate_import_candidate(valid, malware_scanner=lambda path: True)
    assert accepted["ok"] is True


def test_quarantine_release_requires_approval_and_fresh_clean_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    monkeypatch.setenv("EVA_DATA_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    source = tmp_path / "held.txt"
    source.write_text("held engineering document", encoding="utf-8")
    quarantined = validate_and_register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", malware_scanner=lambda path: False,
    )
    quarantine_id = quarantined["quarantine"]["quarantine_id"]

    with pytest.raises(ValueError, match="Explicit quarantine release approval"):
        release_quarantined(
            quarantine_id, owner_id="owner-1", workspace_id="cad",
            actor_id="owner-1", approved=False,
        )
    with pytest.raises(ValueError, match="configured malware scanner"):
        release_quarantined(
            quarantine_id, owner_id="owner-1", workspace_id="cad",
            actor_id="owner-1", approved=True,
        )

    released = release_quarantined(
        quarantine_id, owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1", approved=True,
        malware_scanner=lambda path: True,
    )
    assert released["status"] == "released"
    assert released["record"]["data_id"] == released["data_id"]
    assert released["quarantine_file_retained"] is True
    assert source.exists()
    assert list_quarantined(owner_id="owner-1", workspace_id="cad") == []
