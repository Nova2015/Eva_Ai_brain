"""Dimensional tolerance stack-up calculations.

Implements worst-case and independent-RSS stacks for scalar dimensions. This is not a
complete GD&T engine and does not replace drawing interpretation or qualified review.
All returned dimensions are normalized to millimetres.
"""
from __future__ import annotations

import math
from typing import Any


_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "thou": 0.0254,
}


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def calculate_tolerance_stack(
    dimensions: list[dict[str, Any]],
    *,
    unit: str = "mm",
) -> dict[str, Any]:
    """Calculate worst-case and RSS stack limits from signed dimension contributors."""
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("At least one dimension contributor is required")
    normalized_unit = unit.strip().lower() if isinstance(unit, str) else ""
    if normalized_unit not in _TO_MM:
        raise ValueError(f"Unsupported unit: {unit!r}")
    factor = _TO_MM[normalized_unit]

    nominal_stack = 0.0
    worst_upper = 0.0
    worst_lower = 0.0
    rss_upper_squares = 0.0
    rss_lower_squares = 0.0
    breakdown = []

    for index, item in enumerate(dimensions):
        if not isinstance(item, dict):
            raise ValueError(f"Dimension {index + 1} must be an object")
        name = str(item.get("name") or f"dimension_{index + 1}")
        nominal = _finite_number(item.get("nominal"), f"{name}.nominal") * factor
        plus = _finite_number(item.get("plus_tolerance", 0.0), f"{name}.plus_tolerance") * factor
        minus = _finite_number(item.get("minus_tolerance", 0.0), f"{name}.minus_tolerance") * factor
        direction = item.get("direction", 1)
        if direction not in (-1, 1):
            raise ValueError(f"{name}.direction must be +1 or -1")
        if plus < 0 or minus < 0:
            raise ValueError(f"{name} tolerances must be non-negative")

        nominal_stack += direction * nominal
        upper = plus if direction == 1 else minus
        lower = minus if direction == 1 else plus
        worst_upper += upper
        worst_lower += lower
        rss_upper_squares += upper * upper
        rss_lower_squares += lower * lower
        breakdown.append({
            "name": name,
            "nominal_mm": nominal,
            "upper_contribution_mm": upper,
            "lower_contribution_mm": lower,
            "direction": direction,
        })

    rss_upper = math.sqrt(rss_upper_squares)
    rss_lower = math.sqrt(rss_lower_squares)
    return {
        "unit": "mm",
        "nominal_stack_mm": nominal_stack,
        "worst_case": {
            "minimum_mm": nominal_stack - worst_lower,
            "maximum_mm": nominal_stack + worst_upper,
            "upper_tolerance_mm": worst_upper,
            "lower_tolerance_mm": worst_lower,
        },
        "rss": {
            "minimum_mm": nominal_stack - rss_lower,
            "maximum_mm": nominal_stack + rss_upper,
            "upper_tolerance_mm": rss_upper,
            "lower_tolerance_mm": rss_lower,
            "assumption": "independent contributors; statistical distributions not verified",
        },
        "contributors": breakdown,
        "engineering_approval": "not_granted",
        "gdt_interpretation": "not_performed",
    }


def evaluate_clearance_fit(
    *,
    hole_nominal: float,
    hole_plus_tolerance: float,
    hole_minus_tolerance: float,
    shaft_nominal: float,
    shaft_plus_tolerance: float,
    shaft_minus_tolerance: float,
    unit: str = "mm",
) -> dict[str, Any]:
    """Evaluate a simple hole-minus-shaft clearance using worst-case and RSS limits."""
    stack = calculate_tolerance_stack(
        [
            {
                "name": "hole",
                "nominal": hole_nominal,
                "plus_tolerance": hole_plus_tolerance,
                "minus_tolerance": hole_minus_tolerance,
                "direction": 1,
            },
            {
                "name": "shaft",
                "nominal": shaft_nominal,
                "plus_tolerance": shaft_plus_tolerance,
                "minus_tolerance": shaft_minus_tolerance,
                "direction": -1,
            },
        ],
        unit=unit,
    )
    worst = stack["worst_case"]
    rss = stack["rss"]
    return {
        **stack,
        "fit_type": (
            "clearance" if worst["minimum_mm"] > 0
            else "interference" if worst["maximum_mm"] < 0
            else "transition"
        ),
        "worst_case_guaranteed_clearance": worst["minimum_mm"] > 0,
        "worst_case_guaranteed_interference": worst["maximum_mm"] < 0,
        "rss_nominal_probability_model": "not_assumed",
        "rss_limits": {"minimum_mm": rss["minimum_mm"], "maximum_mm": rss["maximum_mm"]},
    }
