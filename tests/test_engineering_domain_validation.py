from pathlib import Path

import pytest

from src.engineering import domain_validation


def test_step_header_validation_requires_exchange_markers_and_entities(tmp_path):
    valid = tmp_path / "part.step"
    valid.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('test'),'2;1');\nENDSEC;\n"
        "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    result = domain_validation.validate_step_file(valid)
    assert result["ok"] is True
    assert result["entity_count"] == 1
    assert result["geometry_validation"] == "pending_occt_or_cad_kernel"

    invalid = tmp_path / "invalid.step"
    invalid.write_text("not a STEP exchange file", encoding="utf-8")
    assert domain_validation.validate_step_file(invalid)["ok"] is False


def test_stl_validator_checks_triangles_and_degenerate_geometry(tmp_path):
    valid = tmp_path / "valid.stl"
    valid.write_text(
        "solid triangle\n"
        "facet normal 0 0 1\nouter loop\n"
        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid triangle\n",
        encoding="utf-8",
    )
    result = domain_validation.validate_stl_file(valid)
    assert result["ok"] is True
    assert result["triangle_count"] == 1
    assert result["warnings"]  # planar STL is a warning, not an automatic rejection

    degenerate = tmp_path / "degenerate.stl"
    degenerate.write_text(
        "solid bad\nfacet normal 0 0 1\nouter loop\n"
        "vertex 0 0 0\nvertex 0 0 0\nvertex 0 0 0\n"
        "endloop\nendfacet\nendsolid bad\n",
        encoding="utf-8",
    )
    rejected = domain_validation.validate_stl_file(degenerate)
    assert rejected["ok"] is False
    assert rejected["degenerate_triangle_count"] == 1


def test_gcode_validator_flags_operator_actions_and_rejects_unknown_codes(tmp_path):
    safe = tmp_path / "safe.nc"
    safe.write_text(
        "G21\nG90\nG0 X0 Y0 Z5\nG1 X10 Y0 F100\nM5\nM30\n",
        encoding="utf-8",
    )
    result = domain_validation.validate_gcode_file(safe)
    assert result["ok"] is True
    assert result["machine_execution_allowed"] is False
    assert result["operator_review_required"] is False

    review = tmp_path / "review.nc"
    review.write_text("G21\nG0 X0 Y0\nM3 S1200\nM30\n", encoding="utf-8")
    review_result = domain_validation.validate_gcode_file(review)
    assert review_result["ok"] is True
    assert review_result["operator_review_required"] is True
    assert "M3" in review_result["operator_review_codes"]

    invalid = tmp_path / "invalid.nc"
    invalid.write_text("G65 P1000\nG1 X1\n", encoding="utf-8")
    invalid_result = domain_validation.validate_gcode_file(invalid)
    assert invalid_result["ok"] is False
    assert any("unsupported G-code" in issue for issue in invalid_result["issues"])

    with pytest.raises(PermissionError, match="never starts"):
        domain_validation.validate_gcode_file(safe, machine_start_requested=True)


def test_calculix_input_validator_checks_nodes_elements_steps_loads_and_boundaries(tmp_path):
    valid = tmp_path / "valid.inp"
    valid.write_text(
        "*NODE\n"
        "1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n"
        "*ELEMENT, TYPE=C3D4\n1, 1, 2, 3, 4\n"
        "*STEP\n*STATIC\n*CLOAD\n1, 1, 100.0\n"
        "*BOUNDARY\n1, 1, 3\n*END STEP\n",
        encoding="utf-8",
    )
    result = domain_validation.validate_calculix_input(valid)
    assert result["ok"] is True
    assert result["node_count"] == 4
    assert result["element_count"] == 1
    assert result["solver_execution_performed"] is False

    invalid = tmp_path / "invalid.inp"
    invalid.write_text(
        "*NODE\n1, 0, 0, 0\n*ELEMENT, TYPE=C3D4\n1, 1, 2, 3, 4\n",
        encoding="utf-8",
    )
    rejected = domain_validation.validate_calculix_input(invalid)
    assert rejected["ok"] is False
    assert rejected["missing_node_references"] == [2, 3, 4]


def test_domain_validator_dispatches_by_extension(tmp_path):
    path = tmp_path / "valid.nc"
    path.write_text("G0 X0\nM30\n", encoding="utf-8")
    result = domain_validation.validate_engineering_file(path)
    assert result["validator"] == "gcode_syntax"
