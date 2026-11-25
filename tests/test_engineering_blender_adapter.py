import hashlib
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def _governance_db(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_DATA_GOVERNANCE_DB", str(tmp_path / "governance.db"))


from src.data_governance import get_record, list_audit_events
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
            operations, owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=False
        )

    manifest = blender_adapter.build_blender_script(
        operations, scene_name="Door_Trim", owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
    )
    script_path = Path(manifest["script_path"])
    script_text = script_path.read_text(encoding="utf-8")
    assert manifest["status"] == "prepared_for_review"
    assert manifest["execution_performed"] is False
    assert manifest["machine_start_allowed"] is False
    governed_script = get_record(
        manifest["script_data_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert governed_script["sha256"] == manifest["sha256"]
    events = list_audit_events(
        manifest["script_data_id"], owner_id="owner-1", workspace_id="cad"
    )
    assert events[-1]["action"] == "blender_script_prepared"
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
            [operation], owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
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
            owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
        )
    assert existing.read_bytes() == b"existing user export"


def test_blender_script_api_requires_authentication_and_approval(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from src.engineering import routes

    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    request = routes.BlenderScriptRequest(
        operations=[{"op": "create_cube", "name": "Base", "size": 10}],
        workspace_id="cad",
        approved=False,
    )
    with pytest.raises(HTTPException) as auth_error:
        routes.engineering_build_blender_script(request, user_id="anonymous")
    assert auth_error.value.status_code == 401

    with pytest.raises(HTTPException) as approval_error:
        routes.engineering_build_blender_script(request, user_id="owner-1")
    assert approval_error.value.status_code == 403

    request.approved = True
    result = routes.engineering_build_blender_script(request, user_id="owner-1")
    assert result["script"]["status"] == "prepared_for_review"
    assert result["execution_performed"] is False
    assert result["machine_started"] is False



@pytest.mark.parametrize(
    "operation",
    [
        {"op": {"unexpected": "mapping"}},
        {"op": "create_cube", "name": "BoolCube", "size": True},
        {"op": "create_cube", "name": "Cube", "size": 2,
         "location": [0, False, 0]},
    ],
)
def test_blender_rejects_malformed_operation_and_boolean_numeric_values(
    tmp_path, monkeypatch, operation
):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    with pytest.raises(ValueError):
        blender_adapter.build_blender_script(
            [operation], owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
        )



def test_blender_adapter_supports_plane_torus_and_mesh_modifiers(tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    manifest = blender_adapter.build_blender_script(
        [
            {"op": "create_plane", "name": "TrimSurface", "size": 120},
            {"op": "create_torus", "name": "SealRing", "major_radius": 20,
             "minor_radius": 2, "major_segments": 64, "minor_segments": 16},
            {"op": "add_solidify_modifier", "object": "TrimSurface",
             "thickness": 2.5, "offset": -1},
            {"op": "add_subdivision_modifier", "object": "TrimSurface",
             "levels": 2, "render_levels": 3},
            {"op": "add_mirror_modifier", "object": "TrimSurface",
             "axes": ["x", "y"], "use_clip": True},
        ],
        owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True,
    )
    script = Path(manifest["script_path"]).read_text(encoding="utf-8")
    assert "primitive_plane_add" in script
    assert "primitive_torus_add" in script
    assert "SOLIDIFY" in script
    assert "SUBSURF" in script
    assert "MIRROR" in script
    assert [item["op"] for item in manifest["operations"]] == [
        "create_plane", "create_torus", "add_solidify_modifier",
        "add_subdivision_modifier", "add_mirror_modifier",
    ]


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "create_torus", "name": "BadRing", "major_radius": 2,
         "minor_radius": 2},
        {"op": "create_torus", "name": "BadRing", "major_radius": 5,
         "minor_radius": 1, "major_segments": 12.5},
        {"op": "add_mirror_modifier", "object": "Cube", "axes": ["x", "x"]},
        {"op": "add_solidify_modifier", "object": "Cube", "thickness": 1,
         "offset": 2},
    ],
)
def test_blender_adapter_rejects_unsafe_mesh_parameters(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("EVA_ENGINEERING_OUTPUT_DIR", str(tmp_path / "engineering"))
    operations = [{"op": "create_cube", "name": "Cube", "size": 5}, operation]
    with pytest.raises(ValueError):
        blender_adapter.build_blender_script(
            operations, owner_id="owner-1", workspace_id="cad", owner_id="owner-1", workspace_id="cad", actor_id="owner-1", approved=True
        )
