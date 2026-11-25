import pytest

from src.engineering import dfm, materials


def _approved_material(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    item = materials.register_material(
        name="DFM PP", family="thermoplastic", grade="PP-H",
        source_reference="test-material-source",
        properties={
            "density": {"value": 0.9, "unit": "g/cm3"},
            "elastic_modulus": {"value": 1500, "unit": "MPa"},
            "poisson_ratio": {"value": 0.42, "unit": "-"},
        },
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    materials.approve_material(
        item["material_id"], owner_id="owner-1", workspace_id="cad",
        actor_id="reviewer-1", approved=True,
    )
    return item


def test_injection_molding_dfm_reports_configurable_screening_findings(tmp_path, monkeypatch):
    material = _approved_material(tmp_path, monkeypatch)
    result = dfm.analyze_injection_molding_dfm(
        {
            "nominal_wall_thickness_mm": 2.0,
            "wall_thickness_samples_mm": [1.0, 4.0],
            "draft_angles_deg": [0.2, -1.0],
            "ribs": [{"name": "Rib_A", "thickness_mm": 1.5, "height_mm": 8}],
            "bosses": [{"name": "Boss_A", "wall_thickness_mm": 1.4}],
            "undercuts": [{"name": "Hook_A"}],
            "snap_fits": [{
                "name": "Clip_A", "strain_percent": 8, "allowable_strain_percent": 5,
            }],
        },
        material_id=material["material_id"], owner_id="owner-1",
        workspace_id="cad", actor_id="owner-1",
    )
    codes = {finding["code"] for finding in result["findings"]}
    assert {
        "WALL_THICKNESS_VARIATION", "LOW_DRAFT_ANGLE", "RIB_TOO_THICK",
        "RIB_TOO_TALL", "BOSS_TOO_THICK", "UNDERCUT_REVIEW",
        "SNAP_FIT_STRAIN_EXCEEDED",
    }.issubset(codes)
    assert result["status"] == "review_required"
    assert result["advisory_only"] is True
    assert result["engineering_approval"] == "not_granted"
    events = materials.list_material_audit(
        material["material_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert events[-1]["action"] == "dfm_analysis_completed"
    assert events[-1]["details"]["status"] == "review_required"


def test_injection_molding_dfm_preliminary_pass_is_not_engineering_approval(tmp_path, monkeypatch):
    material = _approved_material(tmp_path, monkeypatch)
    result = dfm.analyze_injection_molding_dfm(
        {
            "nominal_wall_thickness_mm": 2.0,
            "wall_thickness_samples_mm": [1.8, 2.0, 2.1],
            "draft_angles_deg": [1.0, 2.0],
            "ribs": [{"name": "Rib_A", "thickness_mm": 1.0, "height_mm": 4.0}],
            "bosses": [{"name": "Boss_A", "wall_thickness_mm": 1.0}],
            "snap_fits": [{
                "name": "Clip_A", "strain_percent": 3, "allowable_strain_percent": 5,
            }],
        },
        material_id=material["material_id"], owner_id="owner-1",
        workspace_id="cad", actor_id="owner-1",
    )
    assert result["status"] == "preliminary_pass"
    assert result["findings"] == []
    assert result["engineering_approval"] == "not_granted"


def test_injection_molding_dfm_requires_approved_material_and_owner_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_DB", str(tmp_path / "engineering.db"))
    pending = materials.register_material(
        name="Pending PP", family="thermoplastic", grade="PP",
        source_reference="test", properties={"elastic_modulus": {"value": 1000, "unit": "MPa"}},
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1",
        reviewer_id="reviewer-1",
    )
    analysis = {"nominal_wall_thickness_mm": 2, "draft_angles_deg": [1]}
    with pytest.raises(PermissionError, match="approved"):
        dfm.analyze_injection_molding_dfm(
            analysis, material_id=pending["material_id"], owner_id="owner-1",
            workspace_id="cad", actor_id="owner-1",
        )
    with pytest.raises(PermissionError, match="data owner"):
        dfm.analyze_injection_molding_dfm(
            analysis, material_id=pending["material_id"], owner_id="owner-1",
            workspace_id="cad", actor_id="attacker",
        )


def test_injection_molding_dfm_api_is_authenticated_and_material_scoped(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    material = _approved_material(tmp_path, monkeypatch)
    request = routes.InjectionMoldingDFMRequest(
        material_id=material["material_id"],
        workspace_id="cad",
        analysis={"nominal_wall_thickness_mm": 2.0, "draft_angles_deg": [1.0]},
    )
    with pytest.raises(HTTPException) as error:
        routes.engineering_analyze_injection_molding_dfm(request, user_id="anonymous")
    assert error.value.status_code == 401

    result = routes.engineering_analyze_injection_molding_dfm(request, user_id="owner-1")
    assert result["dfm"]["status"] == "preliminary_pass"
    assert result["dfm"]["engineering_approval"] == "not_granted"



def test_injection_molding_dfm_rejects_non_string_rule_names(tmp_path, monkeypatch):
    material = _approved_material(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="rule names must be strings"):
        dfm.analyze_injection_molding_dfm(
            {"nominal_wall_thickness_mm": 2, "draft_angles_deg": [1]},
            material_id=material["material_id"], owner_id="owner-1",
            workspace_id="cad", actor_id="owner-1", rules={1: 0.5},
        )
