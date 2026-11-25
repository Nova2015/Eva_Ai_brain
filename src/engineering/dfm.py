"""Advisory, configurable design-for-manufacturing checks for injection-molded parts."""

from __future__ import annotations

import math
from typing import Any

from .materials import record_material_analysis, resolve_material_for_analysis


_DEFAULT_RULES = {
    "max_wall_thickness_variation_ratio": 1.5,
    "max_rib_wall_ratio": 0.6,
    "max_boss_wall_ratio": 0.6,
    "max_rib_height_wall_ratio": 3.0,
    "min_draft_angle_deg": 0.5,
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _positive_number(value: Any, field: str, *, allow_zero: bool = False) -> float:
    number = _finite_number(value, field)
    invalid = number < 0 if allow_zero else number <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def _number_list(value: Any, field: str, *, allow_zero: bool = False) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [
        _positive_number(item, f"{field}[{index}]", allow_zero=allow_zero)
        for index, item in enumerate(value)
    ]


def _validate_rules(rules: dict[str, Any] | None) -> dict[str, float]:
    result = dict(_DEFAULT_RULES)
    if rules is None:
        return result
    if not isinstance(rules, dict):
        raise ValueError("rules must be a dictionary")
    if any(not isinstance(key, str) for key in rules):
        raise ValueError("DFM rule names must be strings")
    unknown = set(rules) - set(_DEFAULT_RULES)
    if unknown:
        raise ValueError("Unsupported DFM rule overrides: " + ", ".join(sorted(unknown)))
    for key, value in rules.items():
        result[key] = _positive_number(
            value, f"rules.{key}", allow_zero=(key == "min_draft_angle_deg")
        )
    return result


def analyze_injection_molding_dfm(
    analysis: dict[str, Any],
    *,
    material_id: str,
    owner_id: str,
    workspace_id: str,
    actor_id: str,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run configurable screening checks; results are advisory, never certification."""
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a dictionary")
    for field, value in (
        ("material_id", material_id), ("owner_id", owner_id),
        ("workspace_id", workspace_id), ("actor_id", actor_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if actor_id != owner_id:
        raise PermissionError("Only the authenticated data owner may run this DFM analysis")

    material = resolve_material_for_analysis(
        material_id, owner_id=owner_id, workspace_id=workspace_id
    )
    thresholds = _validate_rules(rules)
    wall = _positive_number(
        analysis.get("nominal_wall_thickness_mm"), "nominal_wall_thickness_mm"
    )
    samples = analysis.get("wall_thickness_samples_mm", [])
    sample_values = _number_list(samples, "wall_thickness_samples_mm")
    if not sample_values:
        sample_values = [wall]

    findings: list[dict[str, Any]] = []

    def add(code: str, severity: str, message: str, **details: Any) -> None:
        findings.append({
            "code": code, "severity": severity, "message": message, "details": details,
        })

    variation = max(sample_values) / min(sample_values)
    if not math.isfinite(variation):
        raise ValueError("Wall thickness variation ratio overflowed")
    if variation > thresholds["max_wall_thickness_variation_ratio"]:
        add(
            "WALL_THICKNESS_VARIATION", "warning",
            "Wall thickness variation exceeds the configured screening ratio",
            min_mm=min(sample_values), max_mm=max(sample_values),
            ratio=variation, limit=thresholds["max_wall_thickness_variation_ratio"],
        )

    raw_draft_angles = analysis.get("draft_angles_deg", [])
    if not isinstance(raw_draft_angles, list):
        raise ValueError("draft_angles_deg must be a list")
    draft_angles = [
        _finite_number(angle, f"draft_angles_deg[{index}]")
        for index, angle in enumerate(raw_draft_angles)
    ]
    if not draft_angles:
        add("DRAFT_DATA_MISSING", "review", "No draft-angle measurements were supplied")
    else:
        low_draft = [angle for angle in draft_angles if angle < thresholds["min_draft_angle_deg"]]
        if low_draft:
            add(
                "LOW_DRAFT_ANGLE", "warning",
                "One or more draft angles are below the configured screening minimum",
                measured_deg=low_draft, minimum_deg=thresholds["min_draft_angle_deg"],
            )

    ribs = analysis.get("ribs", [])
    if not isinstance(ribs, list):
        raise ValueError("ribs must be a list")
    for index, rib in enumerate(ribs):
        if not isinstance(rib, dict):
            raise ValueError(f"ribs[{index}] must be an object")
        name = rib.get("name", f"rib_{index + 1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ribs[{index}].name must be a non-empty string")
        thickness = _positive_number(rib.get("thickness_mm"), f"{name}.thickness_mm")
        height = _positive_number(rib.get("height_mm"), f"{name}.height_mm")
        thickness_ratio = thickness / wall
        height_ratio = height / wall
        if not math.isfinite(thickness_ratio) or not math.isfinite(height_ratio):
            raise ValueError(f"{name} rib ratios overflowed")
        if thickness_ratio > thresholds["max_rib_wall_ratio"]:
            add(
                "RIB_TOO_THICK", "warning",
                "Rib thickness exceeds the configured wall-thickness ratio",
                feature=name, ratio=thickness_ratio, limit=thresholds["max_rib_wall_ratio"],
            )
        if height_ratio > thresholds["max_rib_height_wall_ratio"]:
            add(
                "RIB_TOO_TALL", "warning",
                "Rib height exceeds the configured wall-thickness ratio",
                feature=name, ratio=height_ratio, limit=thresholds["max_rib_height_wall_ratio"],
            )

    bosses = analysis.get("bosses", [])
    if not isinstance(bosses, list):
        raise ValueError("bosses must be a list")
    for index, boss in enumerate(bosses):
        if not isinstance(boss, dict):
            raise ValueError(f"bosses[{index}] must be an object")
        name = boss.get("name", f"boss_{index + 1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"bosses[{index}].name must be a non-empty string")
        thickness = _positive_number(boss.get("wall_thickness_mm"), f"{name}.wall_thickness_mm")
        ratio = thickness / wall
        if not math.isfinite(ratio):
            raise ValueError(f"{name} boss ratio overflowed")
        if ratio > thresholds["max_boss_wall_ratio"]:
            add(
                "BOSS_TOO_THICK", "warning",
                "Boss wall thickness exceeds the configured base-wall ratio",
                feature=name, ratio=ratio, limit=thresholds["max_boss_wall_ratio"],
            )

    undercuts = analysis.get("undercuts", [])
    if not isinstance(undercuts, list):
        raise ValueError("undercuts must be a list")
    if undercuts:
        add(
            "UNDERCUT_REVIEW", "review",
            "Undercuts require parting-line and side-action review",
            count=len(undercuts),
        )

    snap_fits = analysis.get("snap_fits", [])
    if not isinstance(snap_fits, list):
        raise ValueError("snap_fits must be a list")
    for index, snap in enumerate(snap_fits):
        if not isinstance(snap, dict):
            raise ValueError(f"snap_fits[{index}] must be an object")
        name = snap.get("name", f"snap_fit_{index + 1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"snap_fits[{index}].name must be a non-empty string")
        strain = _positive_number(snap.get("strain_percent"), f"{name}.strain_percent", allow_zero=True)
        allowable = _positive_number(
            snap.get("allowable_strain_percent"), f"{name}.allowable_strain_percent"
        )
        if strain > allowable:
            add(
                "SNAP_FIT_STRAIN_EXCEEDED", "warning",
                "Estimated snap-fit strain exceeds the user-supplied allowable strain",
                feature=name, strain_percent=strain, allowable_percent=allowable,
            )

    status = "review_required" if findings else "preliminary_pass"
    result = {
        "analysis_type": "injection_molding_dfm_screen",
        "ruleset_version": "eva-injection-molding-dfm-0.1",
        "status": status,
        "material_id": material_id,
        "material_name": material["name"],
        "material_family": material["family"],
        "material_grade": material["grade"],
        "material_source_reference": material["source_reference"],
        "nominal_wall_thickness_mm": wall,
        "rules": thresholds,
        "findings": findings,
        "finding_count": len(findings),
        "solver_execution_performed": False,
        "engineering_approval": "not_granted",
        "advisory_only": True,
        "requires_engineer_review": True,
    }
    record_material_analysis(
        material_id,
        owner_id=owner_id,
        workspace_id=workspace_id,
        actor_id=actor_id,
        action="dfm_analysis_completed",
        details={
            "ruleset_version": "eva-injection-molding-dfm-0.1",
            "status": status,
            "finding_codes": [finding["code"] for finding in findings],
            "rules": thresholds,
        },
    )
    return result
