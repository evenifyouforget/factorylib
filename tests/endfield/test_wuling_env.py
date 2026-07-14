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
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_full exactly
    (dollar/rates for the original 12 formulas are unchanged by the dop
    resource split -- ori_to_dop's rate is inserted after xi_sew, and the
    metatransfer is now 25 Dense Originium Powder directly instead of the
    old 50-ori-equivalent; both represent the identical real quantity).

    The 6 trailing zeros are the secondary-goal formulas (Components,
    Sandleaf Powder, Thermal Bank -- see wuling.py's module docstring):
    they're zero-$ so the $-maximizing LP never uses them.
    """
    result = search(preset_1p2e_full())
    assert result.result.status == "optimal"
    assert result.z == 10
    assert np.allclose(result.metatransfer, [0, 0, 0, 0, 0, 0, 0, 0, 25])
    assert np.allclose(
        result.result.formula_rates,
        [
            8,
            393 / 73,
            270,
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
    assert np.allclose(result.result.resource_slack, [0, 0, 0, 0, 0, 0, 0, 0, 0])
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
    assert len(f) == 13


def test_secondary_goals_on_by_default():
    f = build_formulas(WulingConfig())
    assert set(SECONDARY_GOAL_FORMULA_NAMES) <= f.keys()
    assert len(f) == 19


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


def test_sc_and_lc_consume_dop_not_ori_directly():
    """SC/LC Wuling Battery actually consume Dense Originium Powder, a
    resource distinct from raw Originium Ore -- see module docstring."""
    f = build_formulas(WulingConfig())
    dop_index = 8
    ori_index = 1
    assert f["sc"].consumption[dop_index] == 120
    assert f["sc"].consumption[ori_index] == 0
    assert f["lc"].consumption[dop_index] == 90
    assert f["lc"].consumption[ori_index] == 0


def test_components_and_thermal_bank_do_not_consume_dop():
    """Ferrium/Xiranite Component and Thermal Bank use raw Originium Ore
    (via a different refining chain, or directly), not Dense Originium
    Powder -- a metatransfer of DOP must not be spendable by them."""
    f = build_formulas(WulingConfig())
    dop_index = 8
    for name in ("ferrium_component", "xiranite_component", "thermal_bank"):
        assert f[name].consumption[dop_index] == 0


def test_metatransferred_dop_does_not_help_ferrium_component():
    """A DOP-only supply (no local Originium Ore at all) must leave
    ferrium_component unusable: it needs raw Ore (via Origocrust), which
    DOP cannot substitute for. Regression for the fungibility bug where
    "ori" and "dop" were folded into a single resource dimension.

    Forces ferrium_component's output > 0 (it's normally $0) purely to
    make maximize_dollar want to run it if it possibly could, isolating
    whether the resource constraint -- not the $ incentive -- is what's
    actually preventing it."""
    from factorylib.optimize import Formula, maximize_dollar

    f = build_formulas(WulingConfig())["ferrium_component"]
    forced = Formula(consumption=f.consumption, output=1.0, limit=f.limit)
    dop_only_supply = np.array([0, 0, 90, 0, 0, 0, 0, 0, 300], dtype=float)
    result = maximize_dollar(dop_only_supply, [forced])
    assert result.formula_rates[0] == 0.0


def test_dop_metatransfer_still_reproduces_1p2e_full_dollar():
    """The dop-direct metatransfer is mathematically equivalent to the
    old ori-equivalent one for sc/lc's purposes (see module docstring),
    so the search's dollar output is unaffected by the fix."""
    result = search(preset_1p2e_full())
    assert np.isclose(result.result.dollar_output, 206735 / 146)


def test_sandleaf_powder_consumes_nothing():
    f = build_formulas(WulingConfig())
    assert np.allclose(f["sandleaf_powder"].consumption, 0.0)
    assert f["sandleaf_powder"].output == 0
