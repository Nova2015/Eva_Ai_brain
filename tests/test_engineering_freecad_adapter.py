import hashlib
from pathlib import Path

import pytest

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
            operations, actor_id="owner-1", approved=False
        )

    manifest = freecad_adapter.build_freecad_macro(
        operations, document_name="Door_Trim", actor_id="owner-1", approved=True
    )
    macro_path = Path(manifest["macro_path"])
    macro_text = macro_path.read_text(encoding="utf-8")
    assert manifest["status"] == "prepared_for_review"
    assert manifest["execution_performed"] is False
    assert manifest["machine_start_allowed"] is False
    assert manifest["created_by"] == "owner-1"
    assert hashlib.sha256(macro_text.encode("utf-8")).hexdigest() == manifest["sha256"]
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
            [operation], actor_id="owner-1", approved=True
        )


def test_freecad_boolean_cut_requires_existing_objects_and_limits_operation_count(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError, match="earlier objects"):
        freecad_adapter.build_freecad_macro(
            [{"op": "boolean_cut", "name": "Cut", "base": "Base", "tool": "Tool"}],
            actor_id="owner-1", approved=True,
        )
    with pytest.raises(ValueError, match="at most 100"):
        freecad_adapter.build_freecad_macro(
            [{"op": "create_box", "name": f"Box{i}", "length_mm": 1,
              "width_mm": 1, "height_mm": 1} for i in range(101)],
            actor_id="owner-1", approved=True,
        )
