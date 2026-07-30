"""Golden regression tests for `wuling_1p2e.py`.

Reproduces the "headline" $ figures documented across the retired
`tests/wuling/*` Wuling 1.2 -> 1.2d -> 1.2e model lineage (see that
module's own docstring for exactly which historical test each one came
from), purely by varying `build_1p2e_recipes`'s keyword arguments. No
model-version branching is needed, since one resource graph covers all
three by construction.
"""

from __future__ import annotations

import pytest

from factorylib.endfield.wuling_1p2e import build_1p2e_recipes
from factorylib.optimize import solve

_1P2D_SUPPLY = {
    "originium_ore": 540.0,
    "ferrium_ore": 90.0,
    "cuprium_ore": 240.0,
    "max_forges": 12,
}
_1P2_FULL_SUPPLY = {
    "originium_ore": 480.0,
    "ferrium_ore": 90.0,
    "cuprium_ore": 180.0,
    "max_forges": 8,
}


def test_1p2e_full_reproduces_historical_figure() -> None:
    materials, recipes, dollar = build_1p2e_recipes(**_1P2D_SUPPLY)
    result = solve(materials, list(recipes), dollar)
    assert result.status == 0
    assert result.objective == pytest.approx(206735 / 146)

    by_name = {
        recipe.name: rate for recipe, rate in zip(result.recipes, result.multiples)
    }
    expected_rates = {
        "Convert Cuprium Ore": 8.0,
        "React Xiranite with Sewage": 393 / 73,
        "SC Wuling Battery": 59 / 24,
        "LC Wuling Battery": 0.0,
        "Hetonite Part": 13 / 24,
        "Heavy Xiranite": 2.0,
        "Yazhen Syringe A": 11 / 12,
        "Yazhen Syringe C": 0.0,
        "Sell Xiranite": 1350 / 73,
        "Sell Cuprium": 0.0,
        "Purification Building": 393 / 292,
        "Test Area Purification Node": 410 / 73,
    }
    for name, expected_rate in expected_rates.items():
        assert by_name[name] == pytest.approx(expected_rate), name


def test_1p2e_equiv_1p2d_reproduces_historical_figure() -> None:
    """Without the Test Area Purification Node, 1.2e collapses to the
    1.2d baseline exactly."""
    materials, recipes, dollar = build_1p2e_recipes(
        purify_node_max_multiples=0.0, **_1P2D_SUPPLY
    )
    result = solve(materials, list(recipes), dollar)
    assert result.status == 0
    assert result.objective == pytest.approx(2823 / 2)


def test_1p2e_equiv_1p2_full_reproduces_historical_figure() -> None:
    """Without the Test Area Purification Node, at 1.2-full's supply
    numbers, 1.2e collapses to the original Wuling 1.2 baseline
    exactly."""
    materials, recipes, dollar = build_1p2e_recipes(
        purify_node_max_multiples=0.0, **_1P2_FULL_SUPPLY
    )
    result = solve(materials, list(recipes), dollar)
    assert result.status == 0
    assert result.objective == pytest.approx(2229 / 2)
