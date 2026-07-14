import numpy as np
import pytest

from factorylib.endfield.wuling import (
    RESOURCE_NAMES,
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
    split and the de-flattening refactor -- every "_make" formula is a
    lossless, unconstrained pass-through, so the historical $ figure and
    every $-formula's rate are bit-for-bit identical; the metatransfer is
    now 25 Dense Originium Powder directly instead of the old
    50-ori-equivalent, representing the identical real quantity).

    Most of the trailing secondary-goal/plumbing formulas (Components,
    their upstream _make steps, Thermal Bank -- see wuling.py's module
    docstring) are zero-$ dead ends the $-maximizing LP never uses.
    sandleaf_powder is the exception: since ori_to_dop actually consumes
    30 sandleaf/multiple, its rate is pinned to at least ori_to_dop's
    floor demand (9 * 30 / 90 yield = 3.0), not fixed at 0 -- and, being
    $0 itself, is otherwise a genuine LP degeneracy above that floor
    (HiGHS is free to pick anywhere up to its limit=5), so it's checked
    as a range instead of an exact value.
    """
    result = search(preset_1p2e_full())
    assert result.result.status == "optimal"
    assert result.z == 10
    assert np.allclose(
        result.metatransfer,
        [0, 0, 0, 0, 0, 0, 0, 0, 25] + [0] * (len(RESOURCE_NAMES) - 9),
    )
    rates = result.result.formula_rates
    assert np.allclose(
        rates[:19],
        [
            8,  # cup_conv
            393 / 73,  # xi_sew
            9,  # ori_to_dop
            3,  # ferrium_make
            59 / 24,  # xircon_make
            59 / 24,  # sc
            0,  # lc
            13 / 24,  # hetonite_make
            13 / 24,  # hp_make
            13 / 24,  # hp_sell
            2,  # hx_make
            2,  # hx_sell
            11 / 12,  # ya
            0,  # yc
            1350 / 73,  # xi_sell
            0,  # cuprium_part_make
            0,  # cp_sell
            393 / 292,  # purify
            410 / 73,  # purify_node
        ],
    )
    assert np.allclose(rates[19:25], [0, 0, 0, 0, 0, 0])  # the 4 Components + their
    # upstream origocrust_make/packed_origocrust_make
    assert 3.0 - 1e-9 <= rates[25] <= 5.0 + 1e-9  # sandleaf_powder
    assert rates[26] == 0  # thermal_bank
    slack = result.result.resource_slack
    assert np.allclose(slack[:9], [0] * 9)
    assert slack[9] >= -1e-9  # sandleaf surplus, if any
    assert np.allclose(slack[10:], [0] * (len(RESOURCE_NAMES) - 10))
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
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_hp].
    "hp" no longer exists as a single formula (see wuling.py's module
    docstring); banning the $-bearing hp_sell step has the same effect,
    since hetonite_make/hp_make have no other reason to run."""
    config = WulingConfig(formula_limits={"hp_sell": 0})
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


def test_secondary_goals_off_drops_the_goal_formulas_and_their_upstream_make_steps():
    f = build_formulas(WulingConfig(secondary_goals=False))
    assert set(SECONDARY_GOAL_FORMULA_NAMES).isdisjoint(f)
    assert "origocrust_make" not in f  # only ferrium_component needs it
    assert "packed_origocrust_make" not in f  # only xiranite_component needs it
    assert len(f) == 19  # 17 core + purify + purify_node


def test_secondary_goals_on_by_default():
    f = build_formulas(WulingConfig())
    assert set(SECONDARY_GOAL_FORMULA_NAMES) <= f.keys()
    assert (
        len(f) == 27
    )  # 19 + 6 goal formulas + origocrust_make + packed_origocrust_make


def test_secondary_goals_never_change_1p2e_full_dollar_output():
    with_secondary = search(WulingConfig())
    without_secondary = search(WulingConfig(secondary_goals=False))
    assert np.isclose(
        with_secondary.result.dollar_output, without_secondary.result.dollar_output
    )
    assert np.isclose(with_secondary.result.dollar_output, 206735 / 146)


def test_hetonite_component_matches_recipe_line_exactly():
    """12 Hetonite Part + 12 Heavy Xiranite -> 6 Hetonite Component,
    corrected from the raw recipe list's apparent typo "-> 6 Hetonite
    Part" -- now that Hetonite Part/Heavy Xiranite are their own resource
    dimensions, this is an exact 1:1 match to the recipe line instead of
    a scaled collapsed vector."""
    f = build_formulas(WulingConfig())
    consumption = f["hetonite_component"].consumption
    hetonite_part_index = RESOURCE_NAMES.index("hetonite_part")
    heavy_xiranite_index = RESOURCE_NAMES.index("heavy_xiranite")
    assert consumption[hetonite_part_index] == 12.0
    assert consumption[heavy_xiranite_index] == 12.0
    nonzero = {i for i, v in enumerate(consumption) if abs(v) > 1e-12}
    assert nonzero == {hetonite_part_index, heavy_xiranite_index}


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
    """A DOP-only supply (no local Originium Ore or Ferrium Ore at all)
    must leave ferrium_component unusable: it needs refined Origocrust +
    Ferrium (via origocrust_make/ferrium_make's raw-ore refining, not
    Dense Originium Powder). Regression for the fungibility bug where
    "ori" and "dop" were folded into a single resource dimension.

    Includes the real upstream chain (origocrust_make, ferrium_make) and
    forces ferrium_component's output > 0 (it's normally $0) purely to
    make maximize_dollar want to run it if it possibly could, isolating
    whether the resource constraint -- not the $ incentive -- is what's
    actually preventing it."""
    from factorylib.optimize import Formula, maximize_dollar

    f = build_formulas(WulingConfig())
    forced_ferrium_component = Formula(
        consumption=f["ferrium_component"].consumption,
        output=1.0,
        limit=f["ferrium_component"].limit,
    )
    chain = [f["origocrust_make"], f["ferrium_make"], forced_ferrium_component]
    dop_only_supply = np.zeros(len(RESOURCE_NAMES))
    dop_only_supply[RESOURCE_NAMES.index("dop")] = 300.0
    result = maximize_dollar(dop_only_supply, chain)
    assert result.formula_rates[-1] == 0.0


def test_dop_metatransfer_still_reproduces_1p2e_full_dollar():
    """The dop-direct metatransfer is mathematically equivalent to the
    old ori-equivalent one for sc/lc's purposes (see module docstring),
    so the search's dollar output is unaffected by the fix."""
    result = search(preset_1p2e_full())
    assert np.isclose(result.result.dollar_output, 206735 / 146)


def test_sandleaf_powder_produces_sandleaf_consumes_no_base_resource():
    """Fixed from the earlier dead-end bug: Sandleaf Powder is a real
    shared co-input to Grinding Unit recipes (ori_to_dop among them, see
    module docstring), not an unconsumed dead end -- but it still doesn't
    consume any of the *base* tracked resources itself."""
    f = build_formulas(WulingConfig())
    sandleaf_index = 9
    consumption = f["sandleaf_powder"].consumption
    assert np.allclose(consumption[:sandleaf_index], 0.0)
    assert consumption[sandleaf_index] == -90.0
    assert f["sandleaf_powder"].output == 0


def test_ori_to_dop_consumes_sandleaf_when_secondary_goals_on():
    sandleaf_index = 9
    f = build_formulas(WulingConfig())
    assert f["ori_to_dop"].consumption[sandleaf_index] == 30.0


def test_ori_to_dop_ignores_sandleaf_when_secondary_goals_off():
    """Disabling secondary_goals must never change $-optimal search()
    results (see test_secondary_goals_never_change_1p2e_full_dollar_output):
    since nothing produces "sandleaf" when secondary_goals is off,
    ori_to_dop reverts to not tracking that co-input at all, exactly as
    it did before this fix."""
    sandleaf_index = 9
    f = build_formulas(WulingConfig(secondary_goals=False))
    assert f["ori_to_dop"].consumption[sandleaf_index] == 0.0
