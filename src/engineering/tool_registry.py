"""Local-first registry of engineering software and MCP adapters.

This registry is a catalogue and policy gate, not a command runner. No executable,
MCP server, cloud service, CNC controller, or machine is started from this module.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "eva_engineering.db"


def _db_path() -> Path:
    return Path(os.getenv("EVA_ENGINEERING_DB", str(DEFAULT_DB_PATH))).expanduser()

_DEFAULT_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "tool_id": "freecad",
        "name": "FreeCAD",
        "category": "cad",
        "integration_type": "python_api",
        "license_type": "open_source",
        "capabilities": ["parametric_cad", "part_design", "assembly", "drawing", "path_cam"],
        "executable_candidates": ["FreeCAD.exe", "freecad", "FreeCAD"],
        "machine_start_allowed": False,
        "notes": "Primary parametric CAD candidate. Local execution only; not yet validated.",
    },
    {
        "tool_id": "opencascade",
        "name": "Open CASCADE Technology",
        "category": "geometry_kernel",
        "integration_type": "library",
        "license_type": "open_source",
        "capabilities": ["brep", "geometry_kernel", "step", "iges", "boolean_operations"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Geometry kernel candidate; verify installed bindings and license obligations.",
    },
    {
        "tool_id": "catia",
        "name": "CATIA",
        "category": "cad",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["parametric_cad", "surface_design", "automotive_trim", "class_a"],
        "executable_candidates": ["CNEXT.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial candidate. License and automation permissions must be verified.",
    },
    {
        "tool_id": "siemens_nx",
        "name": "Siemens NX",
        "category": "cad",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["parametric_cad", "surface_design", "mold_design", "nx_cam"],
        "executable_candidates": ["ugraf.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial candidate. License and automation permissions must be verified.",
    },
    {
        "tool_id": "solidworks",
        "name": "SOLIDWORKS",
        "category": "cad",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["parametric_cad", "assembly", "drawing", "simulation"],
        "executable_candidates": ["SLDWORKS.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial candidate. License and automation permissions must be verified.",
    },
    {
        "tool_id": "solid_edge",
        "name": "Solid Edge",
        "category": "cad",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["parametric_cad", "sheet_metal", "assembly", "drawing"],
        "executable_candidates": ["Edge.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial candidate. License and automation permissions must be verified.",
    },
    {
        "tool_id": "blender",
        "name": "Blender",
        "category": "3d_content",
        "integration_type": "python_api",
        "license_type": "open_source",
        "capabilities": ["mesh_modeling", "sculpting", "uv", "materials", "rigging", "animation",
                         "geometry_nodes", "rendering", "compositing", "vfx", "python_automation"],
        "executable_candidates": ["blender.exe", "blender"],
        "machine_start_allowed": False,
        "notes": "Full 3D content subsystem, not a replacement for parametric CAD or engineering validation.",
    },
    {
        "tool_id": "vtk",
        "name": "VTK",
        "category": "visualization",
        "integration_type": "python_library",
        "license_type": "open_source",
        "capabilities": ["mesh_visualization", "scientific_visualization", "post_processing"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Library integration candidate; package availability must be verified.",
    },
    {
        "tool_id": "gmsh",
        "name": "Gmsh",
        "category": "mesh",
        "integration_type": "cli_python_api",
        "license_type": "open_source",
        "capabilities": ["mesh_generation", "tetrahedral_mesh", "surface_mesh", "mesh_quality"],
        "executable_candidates": ["gmsh.exe", "gmsh"],
        "machine_start_allowed": False,
        "notes": "Primary mesh-generation candidate.",
    },
    {
        "tool_id": "netgen",
        "name": "Netgen",
        "category": "mesh",
        "integration_type": "cli_python_api",
        "license_type": "open_source",
        "capabilities": ["mesh_generation", "tetrahedral_mesh", "mesh_quality"],
        "executable_candidates": ["netgen.exe", "netgen"],
        "machine_start_allowed": False,
        "notes": "Alternative mesh-generation candidate.",
    },
    {
        "tool_id": "calculix",
        "name": "CalculiX",
        "category": "fea",
        "integration_type": "cli",
        "license_type": "open_source",
        "capabilities": ["linear_static", "nonlinear_static", "modal", "thermal", "buckling"],
        "executable_candidates": ["ccx.exe", "ccx"],
        "machine_start_allowed": False,
        "notes": "Primary structural FEA candidate; benchmark validation required.",
    },
    {
        "tool_id": "code_aster",
        "name": "Code_Aster",
        "category": "fea",
        "integration_type": "cli_wsl",
        "license_type": "open_source",
        "capabilities": ["nonlinear_fea", "thermal", "modal", "fatigue", "contact"],
        "executable_candidates": ["as_run", "code_aster"],
        "machine_start_allowed": False,
        "notes": "Advanced FEA candidate; WSL or native installation must be verified.",
    },
    {
        "tool_id": "elmer_fem",
        "name": "Elmer FEM",
        "category": "multiphysics",
        "integration_type": "cli",
        "license_type": "open_source",
        "capabilities": ["thermal", "electromagnetic", "fluid", "multiphysics"],
        "executable_candidates": ["ElmerSolver.exe", "ElmerSolver"],
        "machine_start_allowed": False,
        "notes": "Multiphysics candidate; validate each physics module independently.",
    },
    {
        "tool_id": "openfoam",
        "name": "OpenFOAM",
        "category": "cfd",
        "integration_type": "cli_wsl",
        "license_type": "open_source",
        "capabilities": ["cfd", "turbulence", "heat_transfer", "multiphase"],
        "executable_candidates": ["simpleFoam", "foamRun"],
        "machine_start_allowed": False,
        "notes": "CFD candidate; WSL adapter required on Windows hosts.",
    },
    {
        "tool_id": "freecad_path",
        "name": "FreeCAD Path Workbench",
        "category": "cam",
        "integration_type": "python_api",
        "license_type": "open_source",
        "capabilities": ["2d_cam", "2_5d_cam", "3_axis_cam", "toolpath_generation", "postprocessing"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "CAM toolpath generation only; does not start a CNC machine.",
    },
    {
        "tool_id": "mastercam",
        "name": "Mastercam",
        "category": "cam",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["2d_cam", "3_axis_cam", "4_axis_cam", "5_axis_cam", "postprocessing"],
        "executable_candidates": ["Mastercam.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial CAM candidate; license and postprocessor validation required.",
    },
    {
        "tool_id": "fusion_manufacturing",
        "name": "Fusion Manufacturing",
        "category": "cam",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["2d_cam", "3_axis_cam", "4_axis_cam", "5_axis_cam", "simulation"],
        "executable_candidates": ["FusionLauncher.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial CAM candidate; verify local data handling and license terms.",
    },
    {
        "tool_id": "hyper_mill",
        "name": "hyperMILL",
        "category": "cam",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["3_axis_cam", "4_axis_cam", "5_axis_cam", "mold_machining"],
        "executable_candidates": ["hyperMILL.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial CAM candidate; postprocessor and license validation required.",
    },
    {
        "tool_id": "nx_cam",
        "name": "NX CAM",
        "category": "cam",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["3_axis_cam", "4_axis_cam", "5_axis_cam", "machine_simulation"],
        "executable_candidates": ["ugraf.exe"],
        "machine_start_allowed": False,
        "notes": "Commercial CAM candidate; license and postprocessor validation required.",
    },
    {
        "tool_id": "linuxcnc",
        "name": "LinuxCNC",
        "category": "cnc_controller",
        "integration_type": "machine_controller",
        "license_type": "open_source",
        "capabilities": ["machine_status", "controller_interface", "gcode_execution"],
        "executable_candidates": ["linuxcnc"],
        "machine_start_allowed": False,
        "notes": "Controller only, not a CAM generator. Machine start is prohibited by EVA policy.",
    },
    {
        "tool_id": "openinjmoldsim",
        "name": "openInjMoldSim",
        "category": "mold_flow",
        "integration_type": "research_adapter",
        "license_type": "research",
        "capabilities": ["injection_molding", "mold_flow", "filling_simulation"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Research candidate; source, dependencies, and validation status must be reviewed.",
    },
    {
        "tool_id": "imfoam",
        "name": "IMFoam",
        "category": "mold_flow",
        "integration_type": "research_adapter",
        "license_type": "research",
        "capabilities": ["injection_molding", "mold_flow", "openfoam_integration"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Research candidate; source, dependencies, and validation status must be reviewed.",
    },
    {
        "tool_id": "autodesk_moldflow",
        "name": "Autodesk Moldflow",
        "category": "mold_flow",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["injection_molding", "filling", "packing", "cooling", "warpage"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Commercial mold-flow candidate; not assumed installed or licensed.",
    },
    {
        "tool_id": "moldex3d",
        "name": "Moldex3D",
        "category": "mold_flow",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["injection_molding", "filling", "cooling", "warpage", "fiber_orientation"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Commercial mold-flow candidate; not assumed installed or licensed.",
    },
    {
        "tool_id": "sigmasoft",
        "name": "SIGMASOFT",
        "category": "mold_flow",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["injection_molding", "mold_thermal", "process_optimization"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Commercial mold-flow candidate; not assumed installed or licensed.",
    },
    {
        "tool_id": "cadmould",
        "name": "CADMOULD",
        "category": "mold_flow",
        "integration_type": "commercial_application",
        "license_type": "commercial",
        "capabilities": ["injection_molding", "filling", "packing", "warpage"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Commercial mold-flow candidate; not assumed installed or licensed.",
    },
    {
        "tool_id": "freecad_mcp",
        "name": "FreeCAD MCP Adapter",
        "category": "mcp",
        "integration_type": "mcp_server",
        "license_type": "unknown",
        "capabilities": ["cad_query", "cad_modeling", "cad_document_operations"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Candidate MCP adapter. Must be source-reviewed, sandboxed, and explicitly approved.",
    },
    {
        "tool_id": "blender_mcp",
        "name": "Blender MCP Adapter",
        "category": "mcp",
        "integration_type": "mcp_server",
        "license_type": "unknown",
        "capabilities": ["blender_scene", "mesh_operations", "rendering", "geometry_nodes"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Candidate MCP adapter. Must be source-reviewed, sandboxed, and explicitly approved.",
    },
    {
        "tool_id": "gmsh_adapter",
        "name": "Gmsh Local Adapter",
        "category": "mcp",
        "integration_type": "local_adapter",
        "license_type": "open_source",
        "capabilities": ["mesh_generation", "mesh_quality", "mesh_export"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Local adapter candidate; no unrestricted MCP tool execution.",
    },
    {
        "tool_id": "calculix_adapter",
        "name": "CalculiX Local Adapter",
        "category": "mcp",
        "integration_type": "local_adapter",
        "license_type": "open_source",
        "capabilities": ["fea_submit", "fea_status", "result_extract"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "Local adapter candidate; benchmark and input validation required.",
    },
    {
        "tool_id": "openfoam_wsl_adapter",
        "name": "OpenFOAM WSL Adapter",
        "category": "mcp",
        "integration_type": "wsl_adapter",
        "license_type": "open_source",
        "capabilities": ["cfd_submit", "cfd_status", "result_extract"],
        "executable_candidates": [],
        "machine_start_allowed": False,
        "notes": "WSL adapter candidate; local-only data flow and command allowlists required.",
    },
)


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS engineering_tools (
            tool_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            integration_type TEXT NOT NULL,
            license_type TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            executable_candidates_json TEXT NOT NULL,
            executable_path TEXT,
            version TEXT,
            validation_status TEXT NOT NULL DEFAULT 'not_checked',
            enabled INTEGER NOT NULL DEFAULT 0,
            license_confirmed INTEGER NOT NULL DEFAULT 0,
            machine_start_allowed INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            updated_by TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    for spec in _DEFAULT_TOOLS:
        conn.execute(
            """INSERT OR IGNORE INTO engineering_tools
               (tool_id, name, category, integration_type, license_type,
                capabilities_json, executable_candidates_json, machine_start_allowed, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                spec["tool_id"], spec["name"], spec["category"],
                spec["integration_type"], spec["license_type"],
                json.dumps(spec["capabilities"], sort_keys=True),
                json.dumps(spec["executable_candidates"]),
                int(bool(spec["machine_start_allowed"])), spec["notes"],
            ),
        )
    conn.commit()


def _dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    item = dict(row)
    item["capabilities"] = json.loads(item.pop("capabilities_json"))
    item["executable_candidates"] = json.loads(item.pop("executable_candidates_json"))
    item["enabled"] = bool(item["enabled"])
    item["license_confirmed"] = bool(item["license_confirmed"])
    item["machine_start_allowed"] = bool(item["machine_start_allowed"])
    return item


def list_tools(
    *, category: Optional[str] = None, enabled_only: bool = False
) -> list[dict[str, Any]]:
    with _connect() as conn:
        _init(conn)
        query = "SELECT * FROM engineering_tools"
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category=?")
            params.append(category)
        if enabled_only:
            clauses.append("enabled=1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY category, name"
        rows = conn.execute(query, params).fetchall()
    return [_dict(row) for row in rows]  # type: ignore[list-item]


def get_tool(tool_id: str) -> Optional[dict[str, Any]]:
    if not isinstance(tool_id, str) or not tool_id.strip():
        raise ValueError("tool_id is required")
    with _connect() as conn:
        _init(conn)
        row = conn.execute(
            "SELECT * FROM engineering_tools WHERE tool_id=?", (tool_id,)
        ).fetchone()
    return _dict(row)


def register_tool(
    *,
    tool_id: str,
    name: str,
    category: str,
    integration_type: str,
    license_type: str,
    capabilities: list[str],
    actor_id: str,
    notes: str = "",
    executable_candidates: Optional[list[str]] = None,
    machine_start_allowed: bool = False,
) -> dict[str, Any]:
    values = (tool_id, name, category, integration_type, license_type, actor_id)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("tool_id, name, category, integration_type, license_type and actor_id are required")
    if not isinstance(capabilities, list) or any(
        not isinstance(item, str) or not item.strip() for item in capabilities
    ):
        raise ValueError("capabilities must be a list of non-empty strings")
    if machine_start_allowed:
        raise ValueError("EVA policy prohibits automatic machine start")
    with _connect() as conn:
        _init(conn)
        conn.execute(
            """INSERT INTO engineering_tools
               (tool_id, name, category, integration_type, license_type,
                capabilities_json, executable_candidates_json, machine_start_allowed,
                notes, updated_by, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, CURRENT_TIMESTAMP)""",
            (
                tool_id.strip(), name.strip(), category.strip(), integration_type.strip(),
                license_type.strip(), json.dumps(sorted(set(capabilities))),
                json.dumps(executable_candidates or []), notes, actor_id.strip(),
            ),
        )
    return get_tool(tool_id)


def validate_tool_installation(
    tool_id: str,
    *,
    executable_path: str,
    actor_id: str,
    version: Optional[str] = None,
) -> dict[str, Any]:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")
    if not isinstance(executable_path, str) or not executable_path.strip():
        raise ValueError("executable_path is required")
    path = Path(executable_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Tool executable path must point to an existing file")
    with _connect() as conn:
        _init(conn)
        cursor = conn.execute(
            """UPDATE engineering_tools
               SET executable_path=?, version=?, validation_status='path_verified',
                   updated_by=?, updated_at=CURRENT_TIMESTAMP
               WHERE tool_id=?""",
            (str(path), version, actor_id.strip(), tool_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown engineering tool: {tool_id}")
    return get_tool(tool_id)  # type: ignore[return-value]


def set_tool_enabled(
    tool_id: str,
    *,
    enabled: bool,
    actor_id: str,
    approved: bool,
    license_confirmed: bool = False,
) -> dict[str, Any]:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("actor_id is required")
    if approved is not True:
        raise PermissionError("Explicit approval is required to change tool enablement")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")

    with _connect() as conn:
        _init(conn)
        row = conn.execute(
            "SELECT * FROM engineering_tools WHERE tool_id=?", (tool_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown engineering tool: {tool_id}")
        if enabled:
            if row["validation_status"] != "path_verified":
                raise ValueError("Tool installation must be validated before enabling")
            if row["license_type"] == "commercial" and license_confirmed is not True:
                raise ValueError("Commercial license confirmation is required before enabling")
        conn.execute(
            """UPDATE engineering_tools
               SET enabled=?, license_confirmed=?, updated_by=?, updated_at=CURRENT_TIMESTAMP
               WHERE tool_id=?""",
            (int(enabled), int(bool(license_confirmed)), actor_id.strip(), tool_id),
        )
    return get_tool(tool_id)  # type: ignore[return-value]


def resolve_tool(tool_id: str) -> dict[str, Any]:
    """Return only a validated, explicitly enabled tool; never executes it."""
    tool = get_tool(tool_id)
    if tool is None:
        raise KeyError(f"Unknown engineering tool: {tool_id}")
    if not tool["enabled"] or tool["validation_status"] != "path_verified":
        raise PermissionError(f"Engineering tool is not enabled and validated: {tool_id}")
    executable = tool.get("executable_path")
    if not executable or not Path(executable).is_file():
        raise PermissionError(f"Validated executable is no longer available: {tool_id}")
    if tool["machine_start_allowed"]:
        raise PermissionError("Automatic machine start is prohibited by EVA policy")
    return tool
