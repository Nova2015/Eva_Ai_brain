import sqlite3

import pytest

from src.engineering import materials


def _properties():
    # Values are synthetic test fixtures, not engineering recommendations.
    return {
        "density": {"value": 1.05, "unit": "g/cm3"},
        "elastic_modulus": {"value": 1500, "unit": "MPa"},
        "poisson_ratio": {"value": 0.36, "unit": "-"},
        "tensile_strength": {"value": 28, "unit": "MPa"},
    }


def test_material_requires_source_and_independent_approval_before_analysis(tmp_path, monkeypatch):
    db_path = tmp_path / "engineering.db"
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(db_path))
    material = materials.register_material(
        name="Example PP",
        family="thermoplastic",
        grade="PP-TD20-EXAMPLE",
        manufacturer="Example Manufacturer",
        source_reference="local://test-fixture/material-datasheet",
        properties=_properties(),
        owner_id="owner-1",
        workspace_id="cad",
        actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    assert material["status"] == "pending_review"
    with pytest.raises(PermissionError, match="Only independently reviewed"):
        materials.resolve_material_for_analysis(
            material["material_id"], owner_id="owner-1", workspace_id="cad"
        )
    with pytest.raises(PermissionError, match="assigned material reviewer"):
        materials.approve_material(
            material["material_id"], owner_id="owner-1", workspace_id="cad",
            actor_id="owner-1", approved=True,
        )

    with pytest.raises(PermissionError, match="assigned material reviewer"):
        materials.approve_material(
            material["material_id"], owner_id="owner-1", workspace_id="cad",
            actor_id="unassigned-reviewer", approved=True,
        )

    approved = materials.approve_material(
        material["material_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="reviewer-1", approved=True,
    )
    assert approved["status"] == "approved"
    assert approved["approved_by"] == "reviewer-1"
    resolved = materials.resolve_material_for_analysis(
        material["material_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert resolved["properties"]["elastic_modulus"] == {
        "value": 1500.0, "unit": "MPa"
    }

    events = materials.list_material_audit(
        material["material_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert [event["action"] for event in events] == [
        "material_registered", "material_approved"
    ]
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE material_audit SET action='tampered' WHERE material_id=?",
                (material["material_id"],),
            )
    finally:
        conn.close()


def test_material_registry_rejects_missing_source_and_invalid_properties(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    base = {
        "name": "Example Steel",
        "family": "metal",
        "grade": "S355-EXAMPLE",
        "source_reference": "local://test-fixture/steel-datasheet",
        "properties": _properties(),
        "owner_id": "owner-1",
        "workspace_id": "cad",
        "actor_id": "owner-1",
        "reviewer_id": "reviewer-1",
    }
    missing_source = dict(base, source_reference="")
    with pytest.raises(ValueError, match="source_reference"):
        materials.register_material(**missing_source)

    invalid = dict(base, properties={
        "elastic_modulus": {"value": float("nan"), "unit": "MPa"}
    })
    with pytest.raises(ValueError, match="finite"):
        materials.register_material(**invalid)

    unsupported = dict(base, properties={"magic_strength": {"value": 1, "unit": "MPa"}})
    with pytest.raises(ValueError, match="Unsupported material property"):
        materials.register_material(**unsupported)

    invalid_poisson = dict(base, properties={
        "poisson_ratio": {"value": 0.7, "unit": "-"}
    })
    with pytest.raises(ValueError, match="poisson_ratio"):
        materials.register_material(**invalid_poisson)


    invalid_unit = dict(base, properties={
        "elastic_modulus": {"value": 1500, "unit": "banana"}
    })
    with pytest.raises(ValueError, match="Unsupported unit"):
        materials.register_material(**invalid_unit)


def test_material_records_are_isolated_by_owner_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    material = materials.register_material(
        name="Example Aluminum", family="metal", grade="6061-EXAMPLE",
        source_reference="local://test-fixture/aluminum-datasheet",
        properties={"density": {"value": 2.7, "unit": "g/cm3"}},
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    assert materials.get_material(
        material["material_id"], owner_id="other-user", workspace_id="cad"
    ) is None
    assert materials.list_materials(
        owner_id="owner-1", workspace_id="other-workspace"
    ) == []


def test_material_rejection_requires_independent_reviewer(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    material = materials.register_material(
        name="Example Nylon", family="thermoplastic", grade="PA6-EXAMPLE",
        source_reference="local://test-fixture/nylon-datasheet",
        properties={"density": {"value": 1.14, "unit": "g/cm3"}},
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    with pytest.raises(PermissionError, match="assigned material reviewer"):
        materials.reject_material(
            material["material_id"], owner_id="owner-1", workspace_id="cad",
            actor_id="owner-1", reason="Missing traceable datasheet",
        )
    rejected = materials.reject_material(
        material["material_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="reviewer-1", reason="Missing traceable datasheet",
    )
    assert rejected["status"] == "rejected"
    with pytest.raises(PermissionError, match="Only independently reviewed"):
        materials.resolve_material_for_analysis(
            material["material_id"], owner_id="owner-1", workspace_id="cad"
        )


def test_material_registry_api_is_authenticated_and_owner_scoped(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    with pytest.raises(HTTPException) as error:
        routes.engineering_materials(workspace_id="cad", user_id="anonymous")
    assert error.value.status_code == 401

    created = routes.engineering_register_material(
        workspace_id="cad",
        req=routes.MaterialRegistrationRequest(
            name="Example PET", family="thermoplastic", grade="PET-EXAMPLE",
            source_reference="local://datasheet/pet-example",
            properties={"density": {"value": 1.38, "unit": "g/cm3"}},
            reviewer_id="reviewer-1",
        ),
        user_id="owner-1",
    )
    assert created["material"]["status"] == "pending_review"
    owner_list = routes.engineering_materials(workspace_id="cad", user_id="owner-1")
    other_list = routes.engineering_materials(workspace_id="cad", user_id="other-user")
    assert len(owner_list["materials"]) == 1
    assert other_list["materials"] == []
    with pytest.raises(HTTPException) as unassigned:
        routes.engineering_approve_material(
            created["material"]["material_id"],
            workspace_id="cad",
            req=routes.MaterialApprovalRequest(approved=True),
            user_id="unassigned-reviewer",
        )
    assert unassigned.value.status_code == 404
    review_queue = routes.engineering_material_review_queue(
        workspace_id="cad", user_id="reviewer-1"
    )
    assert review_queue["materials"][0]["material_id"] == created["material"]["material_id"]
    approved = routes.engineering_approve_material(
        created["material"]["material_id"],
        workspace_id="cad",
        req=routes.MaterialApprovalRequest(approved=True),
        user_id="reviewer-1",
    )
    assert approved["status"] == "approved"
