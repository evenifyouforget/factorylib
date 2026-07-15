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
    split and the de-flattening/uncollapsing refactors -- every "_make"/
    "_solution_make"/"_powder_make"/"_bottle_make" formula is a lossless,
    unconstrained pass-through, so the historical $ figure and every
    $-formula's rate are bit-for-bit identical; the metatransfer is now
    25 Dense Originium Powder directly instead of the old 50-ori-
    equivalent, representing the identical real quantity).

    Most of the trailing secondary-goal/plumbing formulas (Components,
    their upstream _make steps, Thermal Bank, SC/HC Valley Battery -- see
    wuling.py's module docstring) are zero-$ dead ends the $-maximizing
    LP never uses. sandleaf_powder/sandleaf_plant are the exception:
    since ori_to_dop actually consumes 30 sandleaf/multiple, sandleaf_powder's
    rate is pinned to at least ori_to_dop's floor demand (9 * 30 / 90
    yield = 3.0), not fixed at 0 -- and, being $0 itself, is otherwise a
    genuine LP degeneracy above that floor (HiGHS is free to pick
    anywhere up to its limit=5), so it's checked as a range instead of an
    exact value.
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
        rates[:34],
        [
            8,  # cup_conv
            18,  # originium_powder_make
            9,  # ori_to_dop
            3,  # ferrium_make
            3,  # ferrium_powder_make
            393 / 73,  # liquid_xiranite_make
            393 / 73,  # xi_sew
            59 / 24,  # xircon_make
            59 / 24,  # sc_make
            59 / 24,  # sc_sell
            0,  # lc_make
            0,  # lc_sell
            13 / 3,  # cuprium_powder_make
            13 / 3,  # cuprium_solution_make
            13 / 12,  # hetonite_solution_make
            13 / 24,  # hetonite_make
            13 / 24,  # hp_make
            13 / 24,  # hp_sell
            2,  # hx_make
            2,  # hx_sell
            0,  # ferrium_part_make
            11 / 12,  # cuprium_bottle_make
            0,  # ferrium_bottle_make
            11 / 12,  # yazhen_solution_make
            0,  # jincao_solution_make
            11 / 12,  # ya
            0,  # yc
            0,  # jincao_tea
            0,  # jincao_drink
            1350 / 73,  # xi_sell
            11 / 6,  # cuprium_part_make
            0,  # cp_sell
            393 / 292,  # purify
            410 / 73,  # purify_node
        ],
    )
    assert np.allclose(rates[34:40], [0] * 6)  # the 4 Components + origocrust_make
    # + packed_origocrust_make
    assert rates[40] == 5.0  # sandleaf_plant
    assert 3.0 - 1e-9 <= rates[41] <= 5.0 + 1e-9  # sandleaf_powder
    assert np.allclose(rates[42:], [0] * (len(rates) - 42))  # thermal_bank + sc/lc
    # _power + the Steel/SC/HC Valley Battery chain
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
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_ya].

    jincao_tea is a perfect economic substitute for ya now (identical
    recipe shape and price -- see module docstring), so banning ya alone
    no longer reduces $-optimal output; banning both is required to
    reproduce the historical figure, which assumed no such substitute
    existed."""
    config = WulingConfig(formula_limits={"ya": 0, "jincao_tea": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 205129 / 146)


def test_banning_ya_alone_is_absorbed_by_jincao_tea():
    config = WulingConfig(formula_limits={"ya": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 206735 / 146)


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
    assert "sandleaf_plant" not in f  # only sandleaf_powder needs it
    assert "dense_ferrium_powder_make" not in f  # only the Steel/HC Valley chain
    assert len(f) == 34  # 32 core + purify + purify_node


def test_secondary_goals_on_by_default():
    f = build_formulas(WulingConfig())
    assert set(SECONDARY_GOAL_FORMULA_NAMES) <= f.keys()
    # 34 core+conditional + 10 goal formulas (SECONDARY_GOAL_FORMULA_NAMES,
    # now including sc_power/lc_power) + 8 plumbing
    # (SECONDARY_PLUMBING_FORMULA_NAMES)
    assert len(f) == 52


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
    """SC/LC Wuling Battery Packaging actually consumes Dense Originium
    Powder, a resource distinct from raw Originium Ore -- see module
    docstring."""
    f = build_formulas(WulingConfig())
    dop_index = 8
    ori_index = 1
    assert f["sc_make"].consumption[dop_index] == 120
    assert f["sc_make"].consumption[ori_index] == 0
    assert f["lc_make"].consumption[dop_index] == 90
    assert f["lc_make"].consumption[ori_index] == 0


def test_sc_and_lc_battery_are_shared_between_sell_and_power():
    """The same physical battery can't be both sold and burned for
    power -- sc_battery/lc_battery must be a shared resource with
    competing consumers (sc_sell vs. sc_power, lc_sell vs. lc_power),
    not a free byproduct of each."""
    f = build_formulas(WulingConfig())
    sc_battery_index = RESOURCE_NAMES.index("sc_battery")
    lc_battery_index = RESOURCE_NAMES.index("lc_battery")
    assert f["sc_make"].consumption[sc_battery_index] == -6
    assert f["sc_sell"].consumption[sc_battery_index] == 6
    assert f["sc_power"].consumption[sc_battery_index] == 1.5
    assert f["lc_make"].consumption[lc_battery_index] == -6
    assert f["lc_sell"].consumption[lc_battery_index] == 6
    assert f["lc_power"].consumption[lc_battery_index] == 1.5


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


def test_sandleaf_powder_produces_sandleaf_from_sandleaf_raw():
    """Fixed from the earlier dead-end bug: Sandleaf Powder is a real
    shared co-input to Grinding Unit recipes (ori_to_dop among them, see
    module docstring), not an unconsumed dead end. It's fed by
    sandleaf_plant (Planting Unit, uncollapsed from sandleaf_powder
    itself) rather than a tracked *base* resource, so it still doesn't
    consume anything from the original 9-resource base model."""
    f = build_formulas(WulingConfig())
    sandleaf_raw_index = RESOURCE_NAMES.index("sandleaf_raw")
    sandleaf_index = RESOURCE_NAMES.index("sandleaf")
    consumption = f["sandleaf_powder"].consumption
    assert consumption[:9].sum() == 0.0
    assert consumption[sandleaf_raw_index] == 30.0
    assert consumption[sandleaf_index] == -90.0
    assert f["sandleaf_powder"].output == 0
    assert f["sandleaf_plant"].consumption[sandleaf_raw_index] == -30.0
    assert f["sandleaf_plant"].limit == 5


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


def test_xircon_and_hetonite_reactions_need_ferrium_powder_not_ferrium():
    """Both real recipes ("Reactor Crucible: 60 Xircon Effluent + 30
    Ferrium Powder -> ...", "... + 30 Ferrium Powder -> 30 Hetonite +
    ...") need the shredded Powder form specifically, distinct from
    refined Ferrium itself (which ferrium_component/yc/ferrium_part_make
    /ferrium_bottle_make need directly) -- see module docstring."""
    f = build_formulas(WulingConfig())
    ferrium_index = RESOURCE_NAMES.index("ferrium")
    ferrium_powder_index = RESOURCE_NAMES.index("ferrium_powder")
    for name in ("xircon_make", "hetonite_make"):
        assert f[name].consumption[ferrium_index] == 0
        assert f[name].consumption[ferrium_powder_index] == 30


def test_jincao_tea_and_drink_match_ya_yc_price_and_shape():
    """Jincao Tea/Drink are new formulas mirroring Yazhen Syringe A/C
    exactly (same recipe shape, same $ price -- see module docstring),
    just fed by jincao_solution instead of yazhen_solution."""
    f = build_formulas(WulingConfig())
    cuprium_part_index = RESOURCE_NAMES.index("cuprium_part")
    cuprium_bottle_index = RESOURCE_NAMES.index("cuprium_bottle")
    jincao_solution_index = RESOURCE_NAMES.index("jincao_solution")
    assert f["jincao_tea"].consumption[cuprium_part_index] == 60
    assert f["jincao_tea"].consumption[cuprium_bottle_index] == 30
    assert f["jincao_tea"].consumption[jincao_solution_index] == 30
    assert f["jincao_tea"].output == f["ya"].output == 22 * 6

    ferrium_part_index = RESOURCE_NAMES.index("ferrium_part")
    ferrium_bottle_index = RESOURCE_NAMES.index("ferrium_bottle")
    assert f["jincao_drink"].consumption[ferrium_part_index] == 60
    assert f["jincao_drink"].consumption[ferrium_bottle_index] == 30
    assert f["jincao_drink"].consumption[jincao_solution_index] == 30
    assert f["jincao_drink"].output == f["yc"].output == 16 * 6


def test_sc_hc_valley_battery_are_zero_dollar_power_only():
    """Unlike SC/LC *Wuling* Battery, there's no Sell recipe for SC/HC
    Valley Battery in the spec -- their only purpose is feeding Thermal
    Bank's more efficient battery -> power route (see module docstring),
    so they (and their thermal_bank_* consumers) must be zero-$ and
    gated behind secondary_goals."""
    from factorylib.endfield.wuling import POWER_YIELD, SECONDARY_GOAL_FORMULA_NAMES

    f = build_formulas(WulingConfig())
    for name in (
        "sc_valley",
        "hc_valley",
        "thermal_bank_sc_valley",
        "thermal_bank_hc_valley",
    ):
        assert f[name].output == 0

    assert "thermal_bank_sc_valley" in SECONDARY_GOAL_FORMULA_NAMES
    assert "thermal_bank_hc_valley" in SECONDARY_GOAL_FORMULA_NAMES
    assert POWER_YIELD["thermal_bank_sc_valley"] == 420.0
    assert POWER_YIELD["thermal_bank_hc_valley"] == 1100.0

    f_off = build_formulas(WulingConfig(secondary_goals=False))
    for name in (
        "sc_valley",
        "hc_valley",
        "thermal_bank_sc_valley",
        "thermal_bank_hc_valley",
        "dense_ferrium_powder_make",
        "steel_make",
        "steel_part_make",
    ):
        assert name not in f_off


def test_lc_valley_battery_not_modeled():
    """LC Valley Battery needs Amethyst Part, which needs Amethyst Ore --
    a base resource this model doesn't track at all (same reason Cryston
    /Amethyst Component aren't modeled either)."""
    f = build_formulas(WulingConfig())
    assert "lc_valley" not in f
    assert "amethyst" not in RESOURCE_NAMES


def test_sc_valley_battery_chain_is_connected():
    """60 Ferrium Part + 90 Originium Powder -> 6 SC Valley Battery, fed
    by the real Ferrium Ore -> Ferrium -> Ferrium Part chain. Forces
    sc_valley's output > 0 (normally $0) purely to make maximize_dollar
    want to run it if it possibly could, isolating whether the resource
    chain connects at all."""
    from factorylib.optimize import Formula, maximize_dollar

    f = build_formulas(WulingConfig())
    chain_names = [
        "ferrium_make",
        "ferrium_part_make",
        "originium_powder_make",
        "sc_valley",
    ]
    chain = [f[name] for name in chain_names]
    chain[-1] = Formula(
        f["sc_valley"].consumption, output=1.0, limit=f["sc_valley"].limit
    )
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ferr")] = 600
    supply[RESOURCE_NAMES.index("ori")] = 600
    result = maximize_dollar(supply, chain)
    assert result.status == "optimal"
    assert result.dollar_output > 0
    assert result.formula_rates[-1] > 0  # sc_valley actually ran


def test_hc_valley_battery_chain_is_connected():
    """60 Steel Part + 90 Dense Originium Powder -> 6 HC Valley Battery,
    fed by the real Ferrium Ore -> Ferrium -> Ferrium Powder -> Dense
    Ferrium Powder (needs Sandleaf Powder too) -> Steel -> Steel Part
    chain, plus Dense Originium Powder from ori_to_dop."""
    from factorylib.optimize import Formula, maximize_dollar

    f = build_formulas(WulingConfig())
    chain_names = [
        "ferrium_make",
        "ferrium_powder_make",
        "dense_ferrium_powder_make",
        "steel_make",
        "steel_part_make",
        "originium_powder_make",
        "ori_to_dop",
        "sandleaf_plant",
        "sandleaf_powder",
        "hc_valley",
    ]
    chain = [f[name] for name in chain_names]
    chain[-1] = Formula(
        f["hc_valley"].consumption, output=1.0, limit=f["hc_valley"].limit
    )
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ferr")] = 600
    supply[RESOURCE_NAMES.index("ori")] = 600
    result = maximize_dollar(supply, chain)
    assert result.status == "optimal"
    assert result.dollar_output > 0
    assert result.formula_rates[-1] > 0  # hc_valley actually ran


def test_diverting_sc_battery_to_power_reduces_achievable_dollar():
    """Regression for the real trade-off the split exists to model: a
    battery burned for power is one that can't also be sold. sc_power
    itself is $0, so maximize_dollar never chooses it voluntarily --
    instead, cap sc_sell's own limit (as if some batteries were forced
    into Thermal Bank) and confirm the achievable $ from the same
    sc_battery supply strictly drops, rather than power being "free"."""
    from factorylib.optimize import Formula, maximize_dollar

    f = build_formulas(WulingConfig())
    chain_names = [
        "ferrium_make",
        "ferrium_powder_make",
        "xircon_make",
        "sc_make",
        "sc_sell",
    ]
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ferr")] = 90
    supply[RESOURCE_NAMES.index("eff")] = 600
    supply[RESOURCE_NAMES.index("dop")] = 600

    baseline = maximize_dollar(supply, [f[name] for name in chain_names])

    capped_chain = [f[name] for name in chain_names]
    capped_chain[-1] = Formula(f["sc_sell"].consumption, f["sc_sell"].output, limit=1.0)
    capped = maximize_dollar(supply, capped_chain)

    assert capped.dollar_output < baseline.dollar_output
