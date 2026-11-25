import hashlib
from pathlib import Path

import pytest

from src.engineering import calculix_adapter, materials


def _approved_material(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    item = materials.register_material(
        name="Test Steel", family="metal", grade="S355-TEST",
        source_reference="local://test-fixture/s355",
        properties={
            "elastic_modulus": {"value": 210, "unit": "GPa"},
            "poisson_ratio": {"value": 0.3, "unit": "-"},
        },
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    return materials.approve_material(
        item["material_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="reviewer-1", approved=True,
    )


def _model(material_id):
    return {
        "model_name": "Bracket_Static",
        "nodes": [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 10, "y": 0, "z": 0},
            {"id": 3, "x": 0, "y": 10, "z": 0},
            {"id": 4, "x": 0, "y": 0, "z": 10},
        ],
        "elements": [{"id": 1, "nodes": [1, 2, 3, 4]}],
        "element_type": "C3D4",
        "material_id": material_id,
        "owner_id": "owner-1",
        "workspace_id": "cad",
        "actor_id": "owner-1",
        "boundary_conditions": [{"node_id": 1, "dof_start": 1, "dof_end": 3, "value": 0}],
        "concentrated_loads": [{"node_id": 2, "dof": 1, "value": 100}],
        "approved": True,
    }


def test_calculix_deck_uses_approved_material_and_passes_structural_validation(tmp_path, monkeypatch):
    material = _approved_material(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    manifest = calculix_adapter.build_calculix_static_deck(**_model(material["material_id"]))
    deck_path = Path(manifest["deck_path"])
    deck = deck_path.read_text(encoding="utf-8")

    assert manifest["status"] == "prepared_for_review"
    assert manifest["solver_execution_performed"] is False
    assert manifest["engineering_approval"] == "not_granted"
    assert manifest["element_type"] == "C3D4"
    assert manifest["validation"]["ok"] is True
    assert "*ELASTIC\n210000, 0.3" in deck
    assert "*BOUNDARY" in deck
    assert "*CLOAD" in deck
    assert "*END STEP" in deck
    assert hashlib.sha256(deck.encode("utf-8")).hexdigest() == manifest["sha256"]


def test_calculix_deck_rejects_unapproved_material_and_missing_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    material = materials.register_material(
        name="Pending Steel", family="metal", grade="PENDING",
        source_reference="local://test-fixture/pending",
        properties={
            "elastic_modulus": {"value": 210, "unit": "GPa"},
            "poisson_ratio": {"value": 0.3, "unit": "-"},
        },
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    with pytest.raises(PermissionError, match="Only independently reviewed"):
        calculix_adapter.build_calculix_static_deck(**_model(material["material_id"]))
    with pytest.raises(PermissionError, match="Explicit approval"):
        calculix_adapter.build_calculix_static_deck(
            **dict(_model(material["material_id"]), approved=False)
        )


def test_calculix_deck_rejects_invalid_mesh_loads_and_owner_mismatch(tmp_path, monkeypatch):
    material = _approved_material(tmp_path, monkeypatch)
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering-output"))
    model = _model(material["material_id"])
    model["elements"] = [{"id": 1, "nodes": [1, 2, 3, 99]}]
    with pytest.raises(ValueError, match="missing nodes"):
        calculix_adapter.build_calculix_static_deck(**model)

    model = _model(material["material_id"])
    model["concentrated_loads"] = []
    with pytest.raises(ValueError, match="concentrated load"):
        calculix_adapter.build_calculix_static_deck(**model)

    model = _model(material["material_id"])
    model["actor_id"] = "other-user"
    with pytest.raises(PermissionError, match="authenticated data owner"):
        calculix_adapter.build_calculix_static_deck(**model)
