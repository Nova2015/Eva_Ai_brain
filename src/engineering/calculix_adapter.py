"""Safe CalculiX linear-static input deck generator.

Builds a limited, validated C3D4/C3D8 deck from structured mesh, load and boundary
inputs plus an independently approved material record. It never starts CalculiX.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .materials import resolve_material_for_analysis
from .domain_validation import validate_calculix_input


_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,50}$")
_ELASTIC_TO_MPA = {
    "Pa": 1e-6,
    "kPa": 1e-3,
    "MPa": 1.0,
    "GPa": 1000.0,
    "N/mm2": 1.0,
    "N/mm^2": 1.0,
}


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_id(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if number <= 0 or str(number) != str(value):
        raise ValueError(f"{field} must be a positive integer")
    return number


def _output_dir() -> Path:
    configured = os.getenv("EVA_ENGINEERING_OUTPUT_DIR")
    root = Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[2] / "data" / "engineering"
    )
    root = root.resolve()
    target = (root / "fea_inputs").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("FEA input directory must remain inside the engineering output root") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_calculix_static_deck(
    *,
    model_name: str,
    nodes: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    element_type: str,
    material_id: str,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    boundary_conditions: list[dict[str, Any]],
    concentrated_loads: list[dict[str, Any]],
    approved: bool,
) -> dict[str, Any]:
    if approved is not True:
        raise PermissionError("Explicit approval is required to prepare a CalculiX input deck")
    if not isinstance(actor_id, str) or not actor_id.strip() or actor_id != owner_id:
        raise PermissionError("Only the authenticated data owner may prepare this FEA deck")
    if not isinstance(model_name, str) or not _MODEL_NAME.fullmatch(model_name):
        raise ValueError("model_name must be a simple identifier")
    if element_type not in {"C3D4", "C3D8"}:
        raise ValueError("Only first-order C3D4 and C3D8 elements are supported")
    expected_nodes = 4 if element_type == "C3D4" else 8
    if not isinstance(nodes, list) or len(nodes) < expected_nodes:
        raise ValueError(f"At least {expected_nodes} nodes are required for {element_type}")
    if not isinstance(elements, list) or not elements:
        raise ValueError("At least one element is required")
    if not isinstance(boundary_conditions, list) or not boundary_conditions:
        raise ValueError("At least one boundary condition is required")
    if not isinstance(concentrated_loads, list) or not concentrated_loads:
        raise ValueError("At least one concentrated load is required")

    material = resolve_material_for_analysis(
        material_id, owner_id=owner_id, workspace_id=workspace_id
    )
    props = material["properties"]
    elastic = props.get("elastic_modulus")
    poisson = props.get("poisson_ratio")
    if not elastic or not poisson:
        raise ValueError("Approved material must include elastic_modulus and poisson_ratio")
    if elastic["unit"] not in _ELASTIC_TO_MPA:
        raise ValueError("Material elastic modulus unit cannot be converted to MPa")
    elastic_mpa = _number(elastic["value"], "elastic_modulus") * _ELASTIC_TO_MPA[elastic["unit"]]
    poisson_value = _number(poisson["value"], "poisson_ratio")
    if elastic_mpa <= 0 or not -1 < poisson_value < 0.5:
        raise ValueError("Material elastic properties are outside the supported isotropic range")

    node_map: dict[int, tuple[float, float, float]] = {}
    for index, node in enumerate(nodes, 1):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be an object")
        node_id = _positive_id(node.get("id"), f"node_{index}.id")
        if node_id in node_map:
            raise ValueError(f"Duplicate node ID: {node_id}")
        coordinates = tuple(
            _number(node.get(axis), f"node_{node_id}.{axis}") for axis in ("x", "y", "z")
        )
        node_map[node_id] = coordinates

    element_ids: set[int] = set()
    normalized_elements = []
    for index, element in enumerate(elements, 1):
        if not isinstance(element, dict):
            raise ValueError(f"Element {index} must be an object")
        element_id = _positive_id(element.get("id"), f"element_{index}.id")
        if element_id in element_ids:
            raise ValueError(f"Duplicate element ID: {element_id}")
        element_ids.add(element_id)
        refs = element.get("nodes")
        if not isinstance(refs, list) or len(refs) != expected_nodes:
            raise ValueError(f"Element {element_id} must reference exactly {expected_nodes} nodes")
        node_ids = [_positive_id(value, f"element_{element_id}.node") for value in refs]
        missing = [value for value in node_ids if value not in node_map]
        if missing:
            raise ValueError(f"Element {element_id} references missing nodes: {missing}")
        normalized_elements.append((element_id, node_ids))

    normalized_boundaries = []
    for index, boundary in enumerate(boundary_conditions, 1):
        node_id = _positive_id(boundary.get("node_id"), f"boundary_{index}.node_id")
        if node_id not in node_map:
            raise ValueError(f"Boundary condition references missing node {node_id}")
        dof_start = _positive_id(boundary.get("dof_start"), f"boundary_{index}.dof_start")
        dof_end = _positive_id(boundary.get("dof_end"), f"boundary_{index}.dof_end")
        if dof_start > dof_end or dof_end > 3:
            raise ValueError("Boundary DOF range must be within 1..3")
        value = _number(boundary.get("value", 0), f"boundary_{index}.value")
        normalized_boundaries.append((node_id, dof_start, dof_end, value))

    normalized_loads = []
    for index, load in enumerate(concentrated_loads, 1):
        node_id = _positive_id(load.get("node_id"), f"load_{index}.node_id")
        if node_id not in node_map:
            raise ValueError(f"Load references missing node {node_id}")
        dof = _positive_id(load.get("dof"), f"load_{index}.dof")
        if dof > 3:
            raise ValueError("Concentrated load DOF must be within 1..3")
        value = _number(load.get("value"), f"load_{index}.value")
        if value == 0:
            raise ValueError("Concentrated load value must be non-zero")
        normalized_loads.append((node_id, dof, value))

    lines = [
        f"** EVA generated linear-static deck: {model_name}",
        "** Unit system: mm, N, MPa",
        "*HEADING",
        model_name,
        "*NODE",
    ]
    for node_id, xyz in sorted(node_map.items()):
        lines.append(f"{node_id}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")
    lines.append(f"*ELEMENT, TYPE={element_type}, ELSET=EALL")
    for element_id, node_ids in normalized_elements:
        lines.append(f"{element_id}, " + ", ".join(str(value) for value in node_ids))
    lines.extend([
        "*MATERIAL, NAME=EVA_MATERIAL",
        "*ELASTIC",
        f"{elastic_mpa:.12g}, {poisson_value:.12g}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=EVA_MATERIAL",
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
    ])
    for node_id, dof_start, dof_end, value in normalized_boundaries:
        lines.append(f"{node_id}, {dof_start}, {dof_end}, {value:.12g}")
    lines.append("*CLOAD")
    for node_id, dof, value in normalized_loads:
        lines.append(f"{node_id}, {dof}, {value:.12g}")
    lines.append("*END STEP")
    deck_text = "\n".join(lines) + "\n"

    output_dir = _output_dir()
    deck_id = "EVA-CCX-" + uuid.uuid4().hex.upper()
    deck_path = output_dir / f"{deck_id}.inp"
    if deck_path.exists():
        raise FileExistsError("Generated FEA input already exists; refusing to overwrite")
    deck_path.write_text(deck_text, encoding="utf-8")
    validation = validate_calculix_input(deck_path)
    if not validation["ok"]:
        deck_path.unlink(missing_ok=True)
        raise ValueError("Generated CalculiX input failed structural validation: " +
                         "; ".join(validation["issues"]))

    digest = hashlib.sha256(deck_text.encode("utf-8")).hexdigest()
    manifest = {
        "deck_id": deck_id,
        "deck_path": str(deck_path.resolve()),
        "sha256": digest,
        "model_name": model_name,
        "material_id": material_id,
        "material_grade": material["grade"],
        "element_type": element_type,
        "node_count": len(node_map),
        "element_count": len(normalized_elements),
        "unit_system": "mm-N-MPa",
        "validation": validation,
        "status": "prepared_for_review",
        "created_by": actor_id,
        "approved_by": actor_id,
        "solver_execution_performed": False,
        "engineering_approval": "not_granted",
    }
    manifest_path = output_dir / f"{deck_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
