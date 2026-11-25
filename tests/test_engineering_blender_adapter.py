import hashlib
from pathlib import Path

import pytest

from src.engineering import blender_adapter


def test_blender_script_generation_is_approved_and_preserves_existing_scene(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    operations = [
        {"op": "create_cube", "name": "Base", "size": 100, "location": [0, 0, 0]},
        {"op": "create_cylinder", "name": "Boss", "radius": 5, "depth": 10,
         "location": [10, 10, 2.5], "vertices": 48},
        {"op": "set_material", "object": "Base", "material": "Plastic_PP",
         "base_color": [0.1, 0.2, 0.3, 1], "metallic": 0, "roughness": 0.6},
        {"op": "add_bevel_modifier", "object": "Base", "width": 0.5, "segments": 3},
        {"op": "export_glb", "objects": ["Base", "Boss"], "filename": "door_trim.glb"},
        {"op": "save_blend", "filename": "door_trim.blend"},
    ]

    with pytest.raises(PermissionError, match="Explicit approval"):
        blender_adapter.build_blender_script(
            operations, actor_id="owner-1", approved=False
        )

    manifest = blender_adapter.build_blender_script(
        operations, scene_name="Door_Trim", actor_id="owner-1", approved=True
    )
    script_path = Path(manifest["script_path"])
    script_text = script_path.read_text(encoding="utf-8")
    assert manifest["status"] == "prepared_for_review"
    assert manifest["execution_performed"] is False
    assert manifest["machine_start_allowed"] is False
    assert manifest["created_by"] == "owner-1"
    assert hashlib.sha256(script_text.encode("utf-8")).hexdigest() == manifest["sha256"]
    assert "primitive_cube_add" in script_text
    assert "primitive_cylinder_add" in script_text
    assert "Principled BSDF" in script_text
    assert "BEVEL" in script_text
    assert "export_scene.gltf" in script_text
    assert "save_as_mainfile" in script_text
    assert "bpy.ops.object.delete" not in script_text
    assert "os.system" not in script_text
    assert "subprocess" not in script_text
    assert not (tmp_path / "engineering" / "exports" / "door_trim.glb").exists()
    assert not (tmp_path / "engineering" / "exports" / "door_trim.blend").exists()


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "execute_python", "code": "import os; os.remove('x')"},
        {"op": "create_cube", "name": "Bad Name", "size": 10},
        {"op": "create_cube", "name": "Cube", "size": -1},
        {"op": "set_material", "object": "Unknown", "material": "Mat"},
        {"op": "export_glb", "objects": ["Unknown"], "filename": "part.glb"},
        {"op": "save_blend", "filename": "../../outside.blend"},
    ],
)
def test_blender_script_rejects_untrusted_operations(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError):
        blender_adapter.build_blender_script(
            [operation], actor_id="owner-1", approved=True
        )


def test_blender_adapter_never_overwrites_existing_export(tmp_path, monkeypatch):
    root = tmp_path / "engineering"
    export_dir = root / "exports"
    export_dir.mkdir(parents=True)
    existing = export_dir / "part.glb"
    existing.write_bytes(b"existing user export")
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(root))

    with pytest.raises(FileExistsError, match="will not overwrite"):
        blender_adapter.build_blender_script(
            [
                {"op": "create_cube", "name": "Cube", "size": 10},
                {"op": "export_glb", "objects": ["Cube"], "filename": "part.glb"},
            ],
            actor_id="owner-1", approved=True,
        )
    assert existing.read_bytes() == b"existing user export"
