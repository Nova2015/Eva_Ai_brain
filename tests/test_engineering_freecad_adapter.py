import hashlib
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def _governance_db(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))


from src.data_governance import get_record, list_audit_events
from src.engineering import freecad_adapter


def test_freecad_macro_generation_is_structured_approved_and_not_executed(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    operations = [
        {"op": "create_box", "name": "Base", "length_mm": 100, "width_mm": 50, "height_mm": 2.5},
        {"op": "create_cylinder", "name": "Boss", "radius_mm": 5, "height_mm": 10,
         "position_mm": [10, 10, 2.5]},
        {"op": "boolean_cut", "name": "CutResult", "base": "Base", "tool": "Boss"},
        {"op": "export_step", "objects": ["CutResult"], "filename": "door_trim.step"},
        {"op": "save_document", "filename": "door_trim.FCStd"},
    ]

    with pytest.raises(PermissionError, match="Explicit approval"):
        freecad_adapter.build_freecad_macro(
            operations, owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=False
        )

    manifest = freecad_adapter.build_freecad_macro(
        operations, document_name="Door_Trim", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
    )
    macro_path = Path(manifest["macro_path"])
    macro_text = macro_path.read_text(encoding="utf-8")
    assert manifest["status"] == "prepared_for_review"
    assert manifest["execution_performed"] is False
    assert manifest["machine_start_allowed"] is False
    assert manifest["created_by"] == "owner-1"
    assert hashlib.sha256(macro_text.encode("utf-8")).hexdigest() == manifest["sha256"]
    governed_macro = get_record(
        manifest["macro_data_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert governed_macro["sha256"] == manifest["sha256"]
    events = list_audit_events(
        manifest["macro_data_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert events[-1]["action"] == "freecad_macro_prepared"
    assert "Part.makeBox" in macro_text
    assert "Part.makeCylinder" in macro_text
    assert "Part::Cut" in macro_text
    assert "Import.export" in macro_text
    assert "doc.saveAs" in macro_text
    assert "os.system" not in macro_text
    assert "subprocess" not in macro_text
    assert not (tmp_path / "engineering" / "exports" / "door_trim.step").exists()
    assert not (tmp_path / "engineering" / "exports" / "door_trim.FCStd").exists()


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "execute_python", "code": "import os; os.remove('x')"},
        {"op": "create_box", "name": "Bad Name", "length_mm": 1, "width_mm": 1, "height_mm": 1},
        {"op": "create_box", "name": "Box", "length_mm": -1, "width_mm": 1, "height_mm": 1},
        {"op": "export_step", "objects": ["Unknown"], "filename": "part.step"},
        {"op": "save_document", "filename": "../../outside.FCStd"},
    ],
)
def test_freecad_macro_rejects_untrusted_or_invalid_operations(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError):
        freecad_adapter.build_freecad_macro(
            [operation], owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
        )


def test_freecad_boolean_cut_requires_existing_objects_and_limits_operation_count(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError, match="earlier objects"):
        freecad_adapter.build_freecad_macro(
            [{"op": "boolean_cut", "name": "Cut", "base": "Base", "tool": "Tool"}],
            owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
        )
    with pytest.raises(ValueError, match="at most 100"):
        freecad_adapter.build_freecad_macro(
            [{"op": "create_box", "name": f"Box{i}", "length_mm": 1,
              "width_mm": 1, "height_mm": 1} for i in range(101)],
            owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
        )


def test_freecad_macro_api_requires_authentication_and_approval(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    request = routes.FreeCADMacroRequest(
        operations=[
            {"op": "create_box", "name": "Base", "length_mm": 10,
             "width_mm": 10, "height_mm": 2},
        ],
        workspace_id="cad",
        approved=False,
    )
    with pytest.raises(HTTPException) as auth_error:
        routes.engineering_build_freecad_macro(request, user_id="anonymous")
    assert auth_error.value.status_code == 401

    with pytest.raises(HTTPException) as approval_error:
        routes.engineering_build_freecad_macro(request, user_id="owner-1")
    assert approval_error.value.status_code == 403

    request.approved = True
    result = routes.engineering_build_freecad_macro(request, user_id="owner-1")
    assert result["macro"]["status"] == "prepared_for_review"
    assert result["execution_performed"] is False
    assert result["machine_started"] is False



@pytest.mark.parametrize(
    "operation",
    [
        {"op": {"unexpected": "mapping"}},
        {"op": "create_box", "name": "BoolBox", "length_mm": True,
         "width_mm": 1, "height_mm": 1},
    ],
)
def test_freecad_macro_rejects_non_string_operations_and_boolean_dimensions(
    tmp_path, monkeypatch, operation
):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError):
        freecad_adapter.build_freecad_macro(
            [operation], owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
        )



def test_freecad_macro_supports_sphere_cone_and_boolean_combinations(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    manifest = freecad_adapter.build_freecad_macro(
        [
            {"op": "create_sphere", "name": "Ball", "radius_mm": 4,
             "center_mm": [1, 2, 3]},
            {"op": "create_cone", "name": "Taper", "radius1_mm": 5,
             "radius2_mm": 2, "height_mm": 12},
            {"op": "boolean_fuse", "name": "Union", "base": "Ball", "tool": "Taper"},
            {"op": "boolean_common", "name": "Overlap", "base": "Ball", "tool": "Taper"},
        ],
        owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
    )
    macro = Path(manifest["macro_path"]).read_text(encoding="utf-8")
    assert "Part.makeSphere" in macro
    assert "Part.makeCone" in macro
    assert "Part::Fuse" in macro
    assert "Part::Common" in macro
    assert [item["op"] for item in manifest["operations"]] == [
        "create_sphere", "create_cone", "boolean_fuse", "boolean_common",
    ]


def test_freecad_macro_rejects_negative_cone_radius_and_boolean_reference_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError, match="non-negative"):
        freecad_adapter.build_freecad_macro(
            [{"op": "create_cone", "name": "Taper", "radius1_mm": 5,
              "radius2_mm": -1, "height_mm": 10}],
            owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
        )
    with pytest.raises(ValueError, match="earlier objects"):
        freecad_adapter.build_freecad_macro(
            [
                {"op": "create_box", "name": "Base", "length_mm": 2,
                 "width_mm": 2, "height_mm": 2},
                {"op": "boolean_fuse", "name": "Union", "base": "Base", "tool": "Missing"},
            ],
            owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
        )
