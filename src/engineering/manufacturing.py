"""Manufacturing process catalogue for EVA Engineering Suite.

The catalogue describes required engineering inputs and checks. It never starts or
controls a physical machine.
"""
from __future__ import annotations

from typing import Any

from . import tool_registry


_PROCESS_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "process_id": "injection_molding",
        "name": "Injection Molding",
        "category": "plastic_manufacturing",
        "required_inputs": ["part_geometry", "material", "wall_thickness", "mold_concept"],
        "validation_checks": ["wall_thickness", "draft", "ribs_bosses", "gate_runner",
                              "cooling", "ejectors", "parting_line", "shrinkage", "warpage"],
        "tool_candidates": ["freecad", "openinjmoldsim", "imfoam", "autodesk_moldflow",
                            "moldex3d", "sigmasoft", "cadmould"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "blow_molding",
        "name": "Blow Molding",
        "category": "plastic_manufacturing",
        "required_inputs": ["part_geometry", "material", "parison_or_preform", "mold_concept"],
        "validation_checks": ["parison_distribution", "wall_thickness_distribution",
                              "pinch_off", "venting", "cooling", "shrinkage", "warpage"],
        "tool_candidates": ["freecad", "calculix", "openfoam"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "thermoforming",
        "name": "Thermoforming / Vacuum Forming",
        "category": "plastic_manufacturing",
        "required_inputs": ["part_geometry", "sheet_material", "sheet_thickness", "tool_geometry"],
        "validation_checks": ["draw_ratio", "sheet_thinning", "temperature_window",
                              "draft", "trim_line", "vacuum_venting", "springback"],
        "tool_candidates": ["freecad", "calculix", "blender"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "compression_molding",
        "name": "Compression Molding",
        "category": "plastic_manufacturing",
        "required_inputs": ["part_geometry", "material", "charge_mass", "mold_geometry"],
        "validation_checks": ["charge_placement", "pressure", "mold_temperature",
                              "cure_cycle", "flash", "shrinkage", "fiber_orientation"],
        "tool_candidates": ["freecad", "calculix", "openfoam"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "rim_rtm",
        "name": "RIM / RTM",
        "category": "plastic_composite_manufacturing",
        "required_inputs": ["part_geometry", "resin_system", "reinforcement", "mold_geometry"],
        "validation_checks": ["mixing_ratio", "flow_front", "venting", "cure_kinetics",
                              "fiber_orientation", "void_risk", "thermal_cycle"],
        "tool_candidates": ["freecad", "openfoam", "elmer_fem", "calculix"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "cnc_machining",
        "name": "CNC Machining",
        "category": "metal_manufacturing",
        "required_inputs": ["part_geometry", "stock_geometry", "material", "machine_profile", "tooling"],
        "validation_checks": ["workholding", "tool_selection", "feeds_speeds",
                              "3_axis_toolpaths", "4_axis_toolpaths", "5_axis_toolpaths",
                              "collision_check", "postprocessor", "gcode_syntax", "operator_review"],
        "tool_candidates": ["freecad_path", "mastercam", "fusion_manufacturing", "hyper_mill",
                            "nx_cam", "gcode_validator", "linuxcnc"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "edm",
        "name": "Electrical Discharge Machining (EDM)",
        "category": "metal_manufacturing",
        "required_inputs": ["part_geometry", "electrode_geometry", "workpiece_material", "machine_profile"],
        "validation_checks": ["electrode_wear", "spark_gap", "flushing", "surface_finish",
                              "dielectric", "burn_risk"],
        "tool_candidates": ["freecad", "calculix", "python_reporting"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "additive_manufacturing",
        "name": "3D Printing / Additive Manufacturing",
        "category": "additive_manufacturing",
        "required_inputs": ["part_geometry", "material", "printer_profile", "build_orientation"],
        "validation_checks": ["overhangs", "supports", "layer_height", "anisotropy",
                              "thermal_warpage", "build_volume", "post_processing"],
        "tool_candidates": ["freecad", "blender", "gmsh", "netgen"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "sheet_metal_stamping",
        "name": "Sheet Metal Stamping",
        "category": "metal_manufacturing",
        "required_inputs": ["part_geometry", "sheet_material", "sheet_thickness", "tooling_geometry"],
        "validation_checks": ["bend_radius", "k_factor", "bend_allowance", "springback",
                              "forming_limit", "wrinkling", "tearing", "blank_development"],
        "tool_candidates": ["freecad", "code_aster", "calculix", "gmsh"],
        "machine_start_allowed": False,
    },
    {
        "process_id": "plastic_part_assembly",
        "name": "Plastic Part Assembly",
        "category": "assembly",
        "required_inputs": ["component_geometry", "material", "assembly_constraints"],
        "validation_checks": ["clips", "snap_fits", "bosses", "heat_staking",
                              "ultrasonic_welding", "adhesive_bonding", "tolerance_stack",
                              "interference", "assembly_sequence", "retention_force"],
        "tool_candidates": ["freecad", "calculix", "blender", "vtk"],
        "machine_start_allowed": False,
    },
)


def list_processes(category: str | None = None) -> list[dict[str, Any]]:
    items = [dict(item) for item in _PROCESS_CATALOG]
    if category:
        items = [item for item in items if item["category"] == category]
    return items


def get_process(process_id: str) -> dict[str, Any]:
    for item in _PROCESS_CATALOG:
        if item["process_id"] == process_id:
            return dict(item)
    raise KeyError(f"Unknown manufacturing process: {process_id}")


def validate_process_request(
    process_id: str,
    *,
    input_manifest: dict[str, Any],
    machine_start_requested: bool = False,
    preflight_tools: bool = False,
) -> dict[str, Any]:
    process = get_process(process_id)
    if not isinstance(input_manifest, dict):
        raise ValueError("input_manifest must be a dictionary")
    if machine_start_requested:
        raise PermissionError("EVA does not start or control physical manufacturing machines")

    missing = [
        field for field in process["required_inputs"]
        if field not in input_manifest or input_manifest[field] in (None, "", [])
    ]
    if missing:
        raise ValueError("Missing required manufacturing inputs: " + ", ".join(missing))

    tools = []
    for tool_id in process["tool_candidates"]:
        tool = tool_registry.get_tool(tool_id)
        if tool is None:
            continue
        tools.append({
            "tool_id": tool_id,
            "enabled": tool["enabled"],
            "validation_status": tool["validation_status"],
            "license_type": tool["license_type"],
        })
        if preflight_tools:
            tool_registry.resolve_tool(tool_id)

    return {
        "valid": True,
        "process_id": process_id,
        "name": process["name"],
        "required_inputs": process["required_inputs"],
        "validation_checks": process["validation_checks"],
        "tool_candidates": tools,
        "machine_start_allowed": False,
        "physical_execution_performed": False,
    }
