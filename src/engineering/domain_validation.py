"""Local format/domain validators for engineering source files.

These checks are structural safeguards, not certified engineering verification.
Solver results, material behavior, mesh convergence, and machine safety still require
validated domain tools and qualified operator review.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import Any


def _result(ok: bool, validator: str, issues: list[str], **details: Any) -> dict[str, Any]:
    return {
        "ok": ok,
        "validator": validator,
        "status": "passed" if ok else "rejected",
        "issues": issues,
        **details,
    }


def validate_step_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    try:
        text = path.read_text(encoding="latin-1", errors="strict")
    except OSError as exc:
        return _result(False, "step_header", [f"Unable to read STEP file: {type(exc).__name__}"])
    upper = text[: min(len(text), 2_000_000)].upper()
    required = ("ISO-10303-21;", "HEADER;", "DATA;", "ENDSEC;", "END-ISO-10303-21;")
    missing = [token for token in required if token not in upper]
    if missing:
        return _result(False, "step_header", ["Missing STEP exchange markers: " + ", ".join(missing)])
    entity_count = len(re.findall(r"(?m)^\s*#\d+\s*=", text))
    if entity_count == 0:
        return _result(False, "step_header", ["STEP DATA section contains no entity instances"])
    return _result(
        True, "step_header", [], entity_count=entity_count,
        geometry_validation="pending_occt_or_cad_kernel",
    )


def _triangle_area(a: tuple[float, float, float],
                   b: tuple[float, float, float],
                   c: tuple[float, float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def validate_stl_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _result(False, "stl_mesh", [f"Unable to read STL: {type(exc).__name__}"])
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    is_binary = False
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            is_binary = True
            offset = 84
            for _ in range(count):
                values = struct.unpack_from("<12fH", data, offset)
                vertices = (
                    (values[3], values[4], values[5]),
                    (values[6], values[7], values[8]),
                    (values[9], values[10], values[11]),
                )
                triangles.append(vertices)
                offset += 50
    if not is_binary:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return _result(False, "stl_mesh", ["STL is neither a valid binary STL nor UTF-8 ASCII STL"])
        if "solid" not in text[:256].lower() or "endsolid" not in text.lower():
            return _result(False, "stl_mesh", ["ASCII STL solid/endsolid markers are missing"])
        vertices = []
        for match in re.finditer(
            r"(?im)^\s*vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
            text,
        ):
            try:
                vertices.append(tuple(float(match.group(i)) for i in (1, 2, 3)))
            except ValueError:
                return _result(False, "stl_mesh", ["ASCII STL contains an invalid vertex"])
        if not vertices or len(vertices) % 3:
            return _result(False, "stl_mesh", ["ASCII STL must contain complete triangle vertex triplets"])
        triangles = [tuple(vertices[i:i + 3]) for i in range(0, len(vertices), 3)]

    if not triangles:
        return _result(False, "stl_mesh", ["STL contains no triangles"])
    all_points = [point for triangle in triangles for point in triangle]
    if any(not math.isfinite(value) for point in all_points for value in point):
        return _result(False, "stl_mesh", ["STL contains NaN or infinite coordinates"])
    degenerate = sum(1 for a, b, c in triangles if _triangle_area(a, b, c) <= 1e-12)
    mins = [min(point[axis] for point in all_points) for axis in range(3)]
    maxs = [max(point[axis] for point in all_points) for axis in range(3)]
    issues = []
    warnings = []
    if degenerate:
        issues.append(f"STL contains {degenerate} degenerate triangle(s)")
    if any(maxs[i] - mins[i] <= 1e-12 for i in range(3)):
        warnings.append("STL has a zero-thickness bounding-box axis; review thin or planar geometry")
    return _result(
        not issues, "stl_mesh", issues,
        warnings=warnings,
        triangle_count=len(triangles),
        degenerate_triangle_count=degenerate,
        bounding_box={"min": mins, "max": maxs},
        binary=is_binary,
        watertightness="not_checked",
        self_intersection="not_checked",
    )


_GCODE_G = {
    0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 28, 40, 41, 42, 43, 49,
    54, 55, 56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92,
}
_GCODE_M = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30}
_GCODE_TOKEN = re.compile(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


def validate_obj_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _result(False, "obj_mesh", [f"Unable to read OBJ: {type(exc).__name__}"])
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    issues: list[str] = []
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            if len(parts) < 4:
                issues.append(f"Line {line_number}: OBJ vertex requires three coordinates")
                continue
            try:
                point = tuple(float(value) for value in parts[1:4])
            except ValueError:
                issues.append(f"Line {line_number}: invalid OBJ vertex coordinate")
                continue
            if any(not math.isfinite(value) for value in point):
                issues.append(f"Line {line_number}: non-finite OBJ vertex coordinate")
                continue
            vertices.append(point)
        elif parts[0] == "f":
            if len(parts) < 4:
                issues.append(f"Line {line_number}: OBJ face requires at least three vertices")
                continue
            indices = []
            for token in parts[1:]:
                try:
                    raw_index = int(token.split("/", 1)[0])
                except ValueError:
                    issues.append(f"Line {line_number}: invalid OBJ face index")
                    indices = []
                    break
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if index < 0 or index >= len(vertices):
                    issues.append(f"Line {line_number}: OBJ face references a missing vertex")
                    indices = []
                    break
                indices.append(index)
            if indices:
                faces.append(indices)
    if not vertices:
        issues.append("OBJ contains no vertices")
    if not faces:
        issues.append("OBJ contains no polygon faces")
    degenerate_triangles = 0
    for face in faces:
        for index in range(1, len(face) - 1):
            if _triangle_area(vertices[face[0]], vertices[face[index]], vertices[face[index + 1]]) <= 1e-12:
                degenerate_triangles += 1
    if degenerate_triangles:
        issues.append(f"OBJ contains {degenerate_triangles} degenerate triangle(s)")
    if vertices:
        mins = [min(point[axis] for point in vertices) for axis in range(3)]
        maxs = [max(point[axis] for point in vertices) for axis in range(3)]
    else:
        mins, maxs = [], []
    return _result(
        not issues, "obj_mesh", issues,
        vertex_count=len(vertices), face_count=len(faces),
        degenerate_triangle_count=degenerate_triangles,
        bounding_box={"min": mins, "max": maxs} if vertices else None,
        manifoldness="not_checked",
    )


def validate_gmsh_msh_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _result(False, "gmsh_msh", [f"Unable to read ASCII Gmsh mesh: {type(exc).__name__}"])
    try:
        format_index = lines.index("$MeshFormat")
        format_parts = lines[format_index + 1].split()
        version = format_parts[0]
        file_type = int(format_parts[1])
    except (ValueError, IndexError):
        return _result(False, "gmsh_msh", ["Gmsh $MeshFormat header is missing or invalid"])
    if not version.startswith("2."):
        return _result(False, "gmsh_msh", ["Only ASCII Gmsh MSH 2.x is currently validated; export as MSH2"])
    if file_type != 0:
        return _result(False, "gmsh_msh", ["Binary Gmsh meshes are not supported by this validator"])

    issues: list[str] = []
    nodes: set[int] = set()
    elements: list[tuple[int, int, list[int]]] = []
    try:
        node_marker = lines.index("$Nodes")
        expected_node_count = int(lines[node_marker + 1])
        node_rows = lines[node_marker + 2:node_marker + 2 + expected_node_count]
        if len(node_rows) != expected_node_count:
            raise ValueError("node section is shorter than its declared count")
        for raw in node_rows:
            parts = raw.split()
            if len(parts) != 4:
                raise ValueError("invalid node row")
            node_id = int(parts[0])
            coords = [float(value) for value in parts[1:4]]
            if any(not math.isfinite(value) for value in coords):
                raise ValueError("non-finite node coordinate")
            nodes.add(node_id)
        element_marker = lines.index("$Elements")
        expected_element_count = int(lines[element_marker + 1])
        element_rows = lines[element_marker + 2:element_marker + 2 + expected_element_count]
        if len(element_rows) != expected_element_count:
            raise ValueError("element section is shorter than its declared count")
        element_node_counts = {
            1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3,
            9: 6, 10: 9, 11: 10, 12: 27, 13: 18, 14: 14, 15: 1,
            16: 8, 17: 20, 18: 15, 19: 13,
        }
        for raw in element_rows:
            parts = [int(value) for value in raw.split()]
            if len(parts) < 4:
                raise ValueError("invalid element row")
            element_id, element_type, tag_count = parts[:3]
            node_ids = parts[3 + tag_count:]
            expected_count = element_node_counts.get(element_type)
            if expected_count is None:
                raise ValueError(f"unsupported Gmsh element type {element_type}")
            if len(node_ids) != expected_count:
                raise ValueError(f"element {element_id} has an invalid node count")
            elements.append((element_id, element_type, node_ids))
    except (ValueError, IndexError) as exc:
        return _result(False, "gmsh_msh", [f"Invalid MSH2 node/element section: {exc}"])

    if not nodes:
        issues.append("Gmsh mesh contains no nodes")
    if not elements:
        issues.append("Gmsh mesh contains no elements")
    missing = sorted({
        node_id for _, _, refs in elements for node_id in refs if node_id not in nodes
    })
    if missing:
        issues.append(f"Gmsh elements reference missing nodes: {missing[:20]}")
    if "$EndNodes" not in lines or "$EndElements" not in lines:
        issues.append("Gmsh node/element end markers are missing")
    return _result(
        not issues, "gmsh_msh", issues,
        node_count=len(nodes), element_count=len(elements),
        declared_node_count=expected_node_count,
        declared_element_count=expected_element_count,
        missing_node_references=missing,
        element_quality="connectivity_checked_only",
        self_intersection="not_checked",
    )


def validate_gcode_file(
    file_path: str | Path, *, machine_start_requested: bool = False
) -> dict[str, Any]:
    if machine_start_requested:
        raise PermissionError("G-code validation never starts or controls a CNC machine")
    path = Path(file_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _result(False, "gcode_syntax", [f"Unable to read UTF-8 G-code: {type(exc).__name__}"])
    issues: list[str] = []
    review_codes: set[str] = set()
    motion_lines = 0
    has_end = False
    allowed_letters = set("NGMXYZABCUVWIJKRFSTPQL")
    for line_number, raw in enumerate(lines, 1):
        line = re.sub(r"\([^)]*\)", "", raw).split(";", 1)[0].strip().upper()
        if not line or line == "%":
            continue
        if "#" in line or re.search(r"\bO\d+", line):
            issues.append(f"Line {line_number}: macro variables and subprogram labels require separate review")
            continue
        matches = list(_GCODE_TOKEN.finditer(line))
        residue = _GCODE_TOKEN.sub("", line)
        residue = re.sub(r"[\s%]", "", residue)
        if residue:
            issues.append(f"Line {line_number}: unrecognized G-code token(s): {residue[:40]}")
        codes = []
        for match in matches:
            letter = match.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                issues.append(f"Line {line_number}: invalid numeric token")
                continue
            if not math.isfinite(value):
                issues.append(f"Line {line_number}: non-finite numeric value")
                continue
            if letter not in allowed_letters:
                issues.append(f"Line {line_number}: unsupported address letter {letter}")
                continue
            if letter in {"G", "M"}:
                if not value.is_integer():
                    issues.append(f"Line {line_number}: decimal {letter}-codes are not supported")
                    continue
                code = int(value)
                if letter == "G" and code not in _GCODE_G:
                    issues.append(f"Line {line_number}: unsupported G-code G{code}")
                elif letter == "M" and code not in _GCODE_M:
                    issues.append(f"Line {line_number}: unsupported M-code M{code}")
                else:
                    codes.append((letter, code))
                    if letter == "M" and code in {3, 4, 6, 7, 8}:
                        review_codes.add(f"M{code}")
                    if letter == "M" and code in {2, 30}:
                        has_end = True
        if any(letter == "G" and code in {0, 1, 2, 3} for letter, code in codes):
            motion_lines += 1
    if motion_lines == 0:
        issues.append("G-code contains no supported motion blocks")
    if not has_end:
        issues.append("G-code is missing M2 or M30 program end")
    return _result(
        not issues, "gcode_syntax", issues,
        motion_lines=motion_lines,
        operator_review_required=bool(review_codes),
        operator_review_codes=sorted(review_codes),
        machine_execution_allowed=False,
    )


_CALCULIX_ELEMENT_NODES = {
    "C3D4": 4, "C3D8": 8, "C3D10": 10, "C3D20": 20,
    "CPS3": 3, "CPS4": 4, "CPS6": 6, "CPS8": 8,
    "S3": 3, "S4": 4, "S6": 6, "S8": 8,
    "B31": 2, "B32": 3, "T3D2": 2, "T3D3": 3,
}


def validate_calculix_input(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _result(False, "calculix_inp", [f"Unable to read CalculiX input: {type(exc).__name__}"])
    nodes: set[int] = set()
    elements: list[tuple[int, list[int]]] = []
    element_ids: set[int] = set()
    in_nodes = False
    in_elements = False
    current_element_type: str | None = None
    expected_node_count: int | None = None
    step_open = False
    has_step = False
    has_end_step = False
    has_boundary = False
    has_load = False
    issues: list[str] = []
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            keyword = line.split(",", 1)[0].strip().upper()
            in_nodes = keyword == "*NODE"
            in_elements = keyword == "*ELEMENT"
            current_element_type = None
            expected_node_count = None
            if in_elements:
                type_match = re.search(r"\bTYPE\s*=\s*([A-Z0-9]+)", line.upper())
                if type_match:
                    current_element_type = type_match.group(1)
                    expected_node_count = _CALCULIX_ELEMENT_NODES.get(current_element_type)
                if expected_node_count is None:
                    issues.append(f"Line {line_number}: unsupported or missing CalculiX element TYPE")
            if keyword == "*STEP":
                if step_open:
                    issues.append(f"Line {line_number}: nested *STEP blocks are not supported")
                step_open = True
                has_step = True
            elif keyword == "*END STEP":
                if not step_open:
                    issues.append(f"Line {line_number}: *END STEP has no matching *STEP")
                step_open = False
                has_end_step = True
            elif keyword == "*BOUNDARY":
                has_boundary = True
            elif keyword in {"*CLOAD", "*DLOAD", "*DSLOAD"}:
                has_load = True
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        try:
            if in_nodes:
                if len(parts) < 4:
                    raise ValueError("node row requires ID and three coordinates")
                node_id = int(parts[0])
                coords = [float(value) for value in parts[1:4]]
                if node_id <= 0:
                    raise ValueError("node ID must be positive")
                if node_id in nodes:
                    raise ValueError(f"duplicate node ID {node_id}")
                if any(not math.isfinite(value) for value in coords):
                    raise ValueError("node coordinates must be finite")
                nodes.add(node_id)
            elif in_elements:
                if len(parts) < 3 or expected_node_count is None:
                    raise ValueError("element row has no supported element type or connectivity")
                element_id = int(parts[0])
                node_ids = [int(value) for value in parts[1:]]
                if element_id <= 0:
                    raise ValueError("element ID must be positive")
                if element_id in element_ids:
                    raise ValueError(f"duplicate element ID {element_id}")
                if len(node_ids) != expected_node_count:
                    raise ValueError(
                        f"{current_element_type} requires {expected_node_count} node IDs"
                    )
                if any(node_id <= 0 for node_id in node_ids):
                    raise ValueError("element node IDs must be positive")
                if len(set(node_ids)) != len(node_ids):
                    raise ValueError("element connectivity contains duplicate node IDs")
                element_ids.add(element_id)
                elements.append((element_id, node_ids))
        except ValueError as exc:
            issues.append(f"Line {line_number}: invalid node or element data ({exc})")
    if not nodes:
        issues.append("No valid *NODE data found")
    if not elements:
        issues.append("No valid *ELEMENT data found")
    if not has_step or not has_end_step or step_open:
        issues.append("A complete *STEP / *END STEP block is required")
    if not has_boundary:
        issues.append("No *BOUNDARY condition found")
    if not has_load:
        issues.append("No supported load block (*CLOAD, *DLOAD or *DSLOAD) found")
    missing_nodes = sorted({
        node_id for _, refs in elements for node_id in refs if node_id not in nodes
    })
    if missing_nodes:
        issues.append(f"Elements reference missing node IDs: {missing_nodes[:20]}")
    return _result(
        not issues, "calculix_inp", issues,
        node_count=len(nodes),
        element_count=len(elements),
        missing_node_references=missing_nodes,
        solver_execution_performed=False,
        engineering_approval="not_granted",
    )


def validate_engineering_file(file_path: str | Path) -> dict[str, Any]:
    ext = Path(file_path).suffix.lower()
    if ext in {".step", ".stp"}:
        return validate_step_file(file_path)
    if ext == ".stl":
        return validate_stl_file(file_path)
    if ext == ".obj":
        return validate_obj_file(file_path)
    if ext == ".msh":
        return validate_gmsh_msh_file(file_path)
    if ext in {".gcode", ".nc", ".tap", ".ngc"}:
        return validate_gcode_file(file_path)
    if ext == ".inp":
        return validate_calculix_input(file_path)
    return _result(
        True, "format_only", [],
        deep_validation="pending_domain_validator",
        note="No domain-specific validator is registered for this extension.",
    )
