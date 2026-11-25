import pytest

from src.data_governance import list_audit_events, register_import
from src.engineering import validation_service


def _register_file(tmp_path, monkeypatch, suffix=".step"):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))
    source = tmp_path / ("part" + suffix)
    if suffix == ".step":
        source.write_text(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('fixture'),'2;1');\nENDSEC;\n"
            "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n",
            encoding="utf-8",
        )
    else:
        source.write_text("simple local source", encoding="utf-8")
    return source, register_import(
        source, source="test", owner_id="owner-1", workspace_id="cad",
        actor_id="owner-1",
    )


def test_governed_validation_uses_data_id_hash_and_domain_validator(tmp_path, monkeypatch):
    source, record = _register_file(tmp_path, monkeypatch)
    result = validation_service.validate_governed_data(
        record["data_id"], owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
    )
    assert result["data_id"] == record["data_id"]
    assert result["validation"]["validator"] == "step_header"
    assert result["validation"]["ok"] is True
    assert result["source_modified"] is False
    assert source.exists()
    events = list_audit_events(
        record["data_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert events[-1]["action"] == "engineering_validation_completed"
    assert events[-1]["details"]["ok"] is True


def test_governed_validation_rejects_owner_spoofing_and_changed_source(tmp_path, monkeypatch):
    source, record = _register_file(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="data owner"):
        validation_service.validate_governed_data(
            record["data_id"], owner_id="owner-1", workspace_id="cad", actor_id="attacker",
        )

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validation_service.validate_governed_data(
            record["data_id"], owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        )


def test_governed_validation_returns_format_only_for_unknown_extensions(tmp_path, monkeypatch):
    _, record = _register_file(tmp_path, monkeypatch, suffix=".unknown")
    result = validation_service.validate_governed_data(
        record["data_id"], owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
    )
    assert result["validation"]["validator"] == "format_only"
    assert result["validation"]["deep_validation"] == "pending_domain_validator"


def test_governed_validation_api_is_authenticated_and_data_id_scoped(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    _, record = _register_file(tmp_path, monkeypatch)
    request = routes.GovernedDataValidationRequest(workspace_id="cad")
    with pytest.raises(HTTPException) as error:
        routes.engineering_validate_governed_data(
            record["data_id"], request, user_id="anonymous"
        )
    assert error.value.status_code == 401

    result = routes.engineering_validate_governed_data(
        record["data_id"], request, user_id="owner-1"
    )
    assert result["result"]["validation"]["ok"] is True

    with pytest.raises(HTTPException) as scope_error:
        routes.engineering_validate_governed_data(
            record["data_id"],
            routes.GovernedDataValidationRequest(workspace_id="other"),
            user_id="owner-1",
        )
    assert scope_error.value.status_code == 403
