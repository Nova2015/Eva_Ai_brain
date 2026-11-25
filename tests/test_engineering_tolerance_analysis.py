import math

import pytest

from src.engineering.tolerance_analysis import (
    calculate_tolerance_stack,
    evaluate_clearance_fit,
)


def test_worst_case_and_rss_stack_respect_dimension_direction():
    result = calculate_tolerance_stack(
        [
            {"name": "A", "nominal": 10, "plus_tolerance": 0.1,
             "minus_tolerance": 0.2, "direction": 1},
            {"name": "B", "nominal": 5, "plus_tolerance": 0.3,
             "minus_tolerance": 0.1, "direction": -1},
        ]
    )
    assert result["nominal_stack_mm"] == pytest.approx(5.0)
    assert result["worst_case"]["minimum_mm"] == pytest.approx(4.5)
    assert result["worst_case"]["maximum_mm"] == pytest.approx(5.2)
    assert result["rss"]["upper_tolerance_mm"] == pytest.approx(math.sqrt(0.02))
    assert result["rss"]["lower_tolerance_mm"] == pytest.approx(math.sqrt(0.13))
    assert result["engineering_approval"] == "not_granted"
    assert result["gdt_interpretation"] == "not_performed"


def test_tolerance_stack_normalizes_inches_to_millimetres():
    result = calculate_tolerance_stack(
        [{"name": "inch_dimension", "nominal": 1, "plus_tolerance": 0.01,
          "minus_tolerance": 0.01}],
        unit="in",
    )
    assert result["nominal_stack_mm"] == pytest.approx(25.4)
    assert result["worst_case"]["maximum_mm"] == pytest.approx(25.654)


def test_hole_shaft_fit_returns_transition_clearance_limits():
    result = evaluate_clearance_fit(
        hole_nominal=10, hole_plus_tolerance=0.1, hole_minus_tolerance=0.1,
        shaft_nominal=9.9, shaft_plus_tolerance=0.05, shaft_minus_tolerance=0.05,
    )
    assert result["fit_type"] == "transition"
    assert result["worst_case"]["minimum_mm"] == pytest.approx(-0.05)
    assert result["worst_case"]["maximum_mm"] == pytest.approx(0.25)
    assert result["worst_case_guaranteed_clearance"] is False


@pytest.mark.parametrize(
    "dimensions,unit",
    [
        ([], "mm"),
        ([{"nominal": 1, "plus_tolerance": -0.1}], "mm"),
        ([{"nominal": float("nan")}], "mm"),
        ([{"nominal": 1, "direction": 0}], "mm"),
        ([{"nominal": 1}], "furlong"),
    ],
)
def test_tolerance_stack_rejects_invalid_inputs(dimensions, unit):
    with pytest.raises(ValueError):
        calculate_tolerance_stack(dimensions, unit=unit)


def test_tolerance_analysis_api_is_authenticated_and_does_not_grant_approval():
    from fastapi import HTTPException
    from src.engineering import routes

    with pytest.raises(HTTPException) as error:
        routes.engineering_tolerance_stack(
            routes.ToleranceStackRequest(
                dimensions=[{"nominal": 10, "plus_tolerance": 0.1, "minus_tolerance": 0.1}]
            ),
            user_id="anonymous",
        )
    assert error.value.status_code == 401

    result = routes.engineering_tolerance_stack(
        routes.ToleranceStackRequest(
            dimensions=[{"nominal": 10, "plus_tolerance": 0.1, "minus_tolerance": 0.1}]
        ),
        user_id="owner-1",
    )
    assert result["analysis"]["nominal_stack_mm"] == pytest.approx(10)
    assert result["engineering_approval"] == "not_granted"



@pytest.mark.parametrize(
    "dimension",
    [
        {"name": "bool", "nominal": True, "plus_tolerance": 0, "minus_tolerance": 0},
        {"name": "bad-direction", "nominal": 10, "direction": True},
        {"name": "bad-name", "name": 42, "nominal": 10},
    ],
)
def test_tolerance_stack_rejects_boolean_and_malformed_dimension_values(dimension):
    with pytest.raises(ValueError):
        tolerance_analysis.calculate_tolerance_stack([dimension])


def test_tolerance_stack_rejects_overflow_after_unit_conversion():
    with pytest.raises(ValueError, match="overflow"):
        tolerance_analysis.calculate_tolerance_stack([
            {"name": "huge", "nominal": 1e308, "plus_tolerance": 0, "minus_tolerance": 0}
        ], unit="m")
