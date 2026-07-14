import numpy as np
import pytest

from factorylib.endfield.wuling import (
    SECONDARY_GOAL_FORMULA_NAMES,
    WulingConfig,
    build_formulas,
    preset_1p2_full,
    preset_1p2e_equiv_1p2d,
    preset_1p2e_full,
    search,
)


def test_replicates_1p2e_full():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_full exactly.

    The 6 trailing zeros are the secondary-goal formulas (Components,
    Sandleaf Powder, Thermal Bank -- see wuling.py's module docstring):
    they're zero-$ so the $-maximizing LP never uses them, and the first
    12 rates/dollar/slack are unchanged from before that model extension.
    """
    result = search(preset_1p2e_full())
    assert result.result.status == "optimal"
    assert result.z == 10
    assert np.allclose(result.metatransfer, [0, 50, 0, 0, 0, 0, 0, 0])
    assert np.allclose(
        result.result.formula_rates,
        [
            8,
            393 / 73,
            59 / 24,
            0,
            13 / 24,
            2,
            11 / 12,
            0,
            1350 / 73,
            0,
            393 / 292,
            410 / 73,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    assert np.allclose(result.result.resource_slack, [0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(result.result.dollar_output, 206735 / 146)


def test_replicates_1p2e_equiv_1p2d():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_equiv_1p2d."""
    result = search(preset_1p2e_equiv_1p2d())
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 2823 / 2)


def test_replicates_1p2_full():
    """Matches tests/wuling/test_baseline.py::test_wuling_1p2_full (via the
    8-resource collapsed model, purify_node off)."""
    result = search(preset_1p2_full())
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 2229 / 2)


def test_formula_limit_override_ban_ya():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_ya]."""
    config = WulingConfig(formula_limits={"ya": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 205129 / 146)


def test_formula_limit_override_ban_hp():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_hp]."""
    config = WulingConfig(formula_limits={"hp": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 103335 / 73)


def test_no_purify_building_variant():
    """Matches test_wuling_1p2e.py::test_1p2e_variants[no_purify_building]."""
    config = WulingConfig(formula_limits={"purify": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 119335 / 88)


def test_unknown_formula_limit_raises():
    with pytest.raises(ValueError, match="nonexistent"):
        WulingConfig(formula_limits={"nonexistent": 0})


def test_unknown_formula_output_raises():
    with pytest.raises(ValueError, match="nonexistent"):
        WulingConfig(formula_outputs={"nonexistent": 0})


def test_bad_base_supply_shape_raises():
    with pytest.raises(ValueError, match="base_supply"):
        WulingConfig(base_supply=[0, 1, 2])


def test_secondary_goals_off_drops_the_six_new_formulas():
    f = build_formulas(WulingConfig(secondary_goals=False))
    assert set(SECONDARY_GOAL_FORMULA_NAMES).isdisjoint(f)
    assert len(f) == 12


def test_secondary_goals_on_by_default():
    f = build_formulas(WulingConfig())
    assert set(SECONDARY_GOAL_FORMULA_NAMES) <= f.keys()
    assert len(f) == 18


def test_secondary_goals_never_change_1p2e_full_dollar_output():
    with_secondary = search(WulingConfig())
    without_secondary = search(WulingConfig(secondary_goals=False))
    assert np.isclose(
        with_secondary.result.dollar_output, without_secondary.result.dollar_output
    )
    assert np.isclose(with_secondary.result.dollar_output, 206735 / 146)


def test_hetonite_component_consumption_matches_2x_hp_plus_2x_hx():
    """12 Hetonite Part + 12 Heavy Xiranite = 2 runs each of hp/hx (each
    produces 6 units per run)."""
    f = build_formulas(WulingConfig())
    expected = 2 * f["hp"].consumption + 2 * f["hx"].consumption
    assert np.allclose(f["hetonite_component"].consumption, expected)


def test_sandleaf_powder_consumes_nothing():
    f = build_formulas(WulingConfig())
    assert np.allclose(f["sandleaf_powder"].consumption, 0.0)
    assert f["sandleaf_powder"].output == 0
