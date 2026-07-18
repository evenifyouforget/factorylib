import numpy as np
import pytest

from factorylib.endfield import wuling as v1p2e
from factorylib.endfield.wuling_1p4 import (
    _NEW_RESOURCE_NAMES,
    _THRESHOLD_RECIPES,
    FORMULA_WATTS,
    GOOD_YIELD,
    RESOURCE_NAMES,
    WulingConfig1p4,
    _default_base_supply,
    build_formulas,
    full_supply,
    power_dollar_tax_paid,
    search,
)


def _new_1p4_formula_names(config: WulingConfig1p4) -> set[str]:
    """Every formula name in wuling_1p4's output that isn't also a name
    in plain 1.2e's own build_formulas() -- computed as a set
    difference rather than hand-maintained, so it can't silently go
    stale as more 1.4 formulas get added. Gases (and every other 1.4-
    specific resource) didn't exist in 1.2e at all and have no base
    supply of their own, so banning every one of these formulas (via
    formula_limits=0) structurally eliminates them entirely -- there's
    no other path to touch them."""
    base_names = set(v1p2e.build_formulas(v1p2e.WulingConfig()).keys())
    p4_names = set(build_formulas(config).keys())
    return p4_names - base_names


def test_build_formulas_does_not_raise():
    formulas = build_formulas(WulingConfig1p4())
    assert len(formulas) > 0


def test_every_new_resource_has_base_supply_or_a_producer():
    """Regression for a real bug: liquid_heavy_xiranite was consumed by
    fluid_gas_heavy_xiragen but no formula produced it and it had no
    base supply either, silently making that recipe permanently
    infeasible. Every new 1.4 resource must be either a base-supplied
    raw material or have at least one formula that produces it (a
    negative consumption entry) -- this is the general form of the
    "two things treated as fungible" resource-fidelity bug class
    wuling.py's own module docstring discusses (DOP/Origocrust), just
    the opposite failure mode (a real distinction with no bridge at all,
    instead of two things wrongly merged into one).

    Pyrrolite is no longer a special case (see
    test_pyrrolite_has_no_base_supply_but_is_reachable_via_reverse_recipe):
    it now has a producer -- solid_gas_pyrrolite_gas_reverse, turning
    Gas Reactor Globe's Pyrrolite Gas back into solid Pyrrolite -- so
    every new resource is checked uniformly here."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    supply = full_supply(config)
    producers: dict[str, list[str]] = {name: [] for name in _NEW_RESOURCE_NAMES}
    for formula_name, formula in formulas.items():
        for resource_name in _NEW_RESOURCE_NAMES:
            idx = RESOURCE_NAMES.index(resource_name)
            if formula.consumption[idx] < 0:
                producers[resource_name].append(formula_name)

    unreachable = [
        name
        for name in _NEW_RESOURCE_NAMES
        if not producers[name] and supply[RESOURCE_NAMES.index(name)] <= 0.0
    ]
    assert unreachable == []


def test_pyrrolite_has_no_base_supply_but_is_reachable_via_reverse_recipe():
    """Pyrrolite must have zero base supply (confirmed with the user:
    it's crafted, not sourced directly) -- but unlike the earlier gap,
    it's no longer unreachable: Gas Reactor Globe makes Pyrrolite Gas
    (2 Hetonite Gas + 1 Xiragen -> 1 Pyrrolite Gas, Acrid ENV), and the
    reverse of Solid-Gas Transmuting Unit's own Pyrrolite recipe turns
    that back into solid Pyrrolite (confirmed with the user: both
    directions of every Fluid-Gas/Solid-Gas Transmuting Unit recipe are
    modeled, mirroring the forward ratio and activation cost with
    input/output swapped)."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    supply = full_supply(config)
    assert supply[RESOURCE_NAMES.index("pyrrolite")] == 0.0
    # Default config models every threshold recipe as the two-layer
    # integer alloc/run pair (see WulingConfig1p4.continuous_thresholds),
    # so the real per-unit consumption lives on "_run", not the bare name.
    assert "solid_gas_pyrrolite_gas_reverse_run" in formulas
    assert (
        formulas["solid_gas_pyrrolite_gas_reverse_run"].consumption[
            RESOURCE_NAMES.index("pyrrolite")
        ]
        < 0
    )
    # The unconstrained $-optimal solution may not happen to choose this
    # path (it now competes with several other reverse recipes for the
    # same scarce Acrid ENV/Hetonite Gas/Xiragen -- an economic choice,
    # not unreachability), so verify genuine feasibility directly: give
    # pyrrolite_part_sell an overwhelming incentive (formula_outputs
    # already supports overriding a formula's $ value) and confirm the
    # LP actually produces some.
    forced_config = WulingConfig1p4(formula_outputs={"pyrrolite_part_sell": 1e9})
    result, names = search(forced_config)
    assert result.status == "optimal"
    rates = dict(zip(names, result.formula_rates))
    assert rates["pyrrolite_part_sell"] > 0.0


def test_build_formulas_reuses_1p2e_recipes_unchanged():
    """cup_conv (Cuprium Ore Refining) is untouched by 1.4 -- its
    consumption ratio (30 Cuprium Ore -> 30 Cuprium + 30 Sewage) must
    survive extension to the new, longer resource vector unchanged."""
    formulas = build_formulas(WulingConfig1p4())
    cup_conv = formulas["cup_conv"]
    assert cup_conv.consumption[RESOURCE_NAMES.index("cup_ore")] == 30.0
    assert cup_conv.consumption[RESOURCE_NAMES.index("cup")] == -30.0
    assert cup_conv.consumption[RESOURCE_NAMES.index("sew")] == -30.0


def test_old_forge_allocation_formulas_are_removed():
    formulas = build_formulas(WulingConfig1p4())
    assert "xiranite_forge_alloc" not in formulas
    assert "heavy_xiranite_forge_alloc" not in formulas


def test_xiranite_solid_liquid_gas_forms_are_distinct_resources():
    """Xiranite (solid, "xi"), Liquid Xiranite ("liquid_xiranite"), and
    Xiragen (gas, "xiragen") are the same underlying material in three
    physical states, but -- like the historical DOP/Origocrust bug
    wuling.py's own module docstring warns about -- must never be
    treated as fungible: every conversion between them needs its own
    real Formula with a genuine cost, not a free reinterpretation of the
    same resource index."""
    assert (
        len({RESOURCE_NAMES.index(n) for n in ("xi", "liquid_xiranite", "xiragen")})
        == 3
    )


def test_converting_between_xiranite_forms_is_never_free():
    """Every formula that produces xi, liquid_xiranite, or xiragen must
    also consume something real -- either a genuinely different resource
    (e.g. Stabilized Carbon for xi_forge_run) or one of the *other* two
    forms (e.g. liquid_xiranite_make legitimately spends xi to produce
    liquid_xiranite -- that's a real cost, not a free relabeling, since
    "one of the three forms" is not the same as "nothing")."""
    formulas = build_formulas(WulingConfig1p4())
    forms = {"xi", "liquid_xiranite", "xiragen"}
    for name, formula in formulas.items():
        produces_a_form = any(
            formula.consumption[RESOURCE_NAMES.index(f)] < 0 for f in forms
        )
        if not produces_a_form:
            continue
        real_cost = any(amount > 0 for amount in formula.consumption)
        assert real_cost, f"{name} produces a Xiranite form for free"


def test_heavy_xiranite_solid_liquid_gas_forms_are_distinct_resources():
    """Same fidelity check for Heavy Xiranite's three forms -- this is
    exactly the class of bug caught and fixed while writing this module
    (liquid_heavy_xiranite had no producer at all; see
    test_every_new_resource_has_base_supply_or_a_producer)."""
    assert (
        len(
            {
                RESOURCE_NAMES.index(n)
                for n in ("heavy_xiranite", "liquid_heavy_xiranite", "heavy_xiragen")
            }
        )
        == 3
    )


def test_search_reaches_optimal_status():
    result, names = search(WulingConfig1p4())
    assert result.status == "optimal"
    assert result.dollar_output > 0
    assert len(names) == len(result.formula_rates)


def test_search_result_is_feasible():
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    result, names = search(config)
    consumption = np.stack([formulas[name].consumption for name in names], axis=1)
    supply = full_supply(config)
    assert np.all(consumption @ result.formula_rates <= supply + 1e-6)


def test_hx_make_reused_recipe_matches_new_confirmed_ratio():
    """hx_make (Heavy Xiranite Forge) is reused from 1.2e unchanged --
    confirm its ratio (60 Xiranite + 30 Xircon Effluent -> 6 Heavy
    Xiranite per hx_forge_capacity unit) exactly matches the newly
    confirmed real recipe (10 Xiranite + 5 Xircon Effluent -> 1 Heavy
    Xiranite, i.e. the same 10:5:1 ratio scaled by 6)."""
    formulas = build_formulas(WulingConfig1p4())
    hx_make = formulas["hx_make"]
    xi = hx_make.consumption[RESOURCE_NAMES.index("xi")]
    eff = hx_make.consumption[RESOURCE_NAMES.index("eff")]
    heavy_xiranite = hx_make.consumption[RESOURCE_NAMES.index("heavy_xiranite")]
    assert xi / -heavy_xiranite == pytest.approx(10.0)
    assert eff / -heavy_xiranite == pytest.approx(5.0)


def test_forge_allocations_share_the_same_12_building_budget():
    """xi_forge_alloc, xi_forge_stable_env_alloc, and hx_forge_alloc must
    all draw from the same forge_budget pool, each individually capped
    at max_forges -- confirmed with the user: 12 total buildings, 3
    competing recipes, not 3 independent 12-building pools."""
    config = WulingConfig1p4(max_forges=12)
    formulas = build_formulas(config)
    for name in ("xi_forge_alloc", "xi_forge_stable_env_alloc", "hx_forge_alloc"):
        assert formulas[name].limit == 12.0
        assert formulas[name].integer is True
        assert formulas[name].consumption[RESOURCE_NAMES.index("forge_budget")] == 1.0

    # feasibility: committing all 12 to two of the three should leave no
    # forge_budget for the third once its own 12-unit limit is checked
    # against the pool, not just its own limit
    supply = full_supply(config)
    consumption = np.stack(
        [
            formulas["xi_forge_alloc"].consumption,
            formulas["xi_forge_stable_env_alloc"].consumption,
            formulas["hx_forge_alloc"].consumption,
        ],
        axis=1,
    )
    over_budget_rates = np.array([6.0, 6.0, 6.0])  # 18 total, only 12 available
    usage = consumption @ over_budget_rates
    assert (
        usage[RESOURCE_NAMES.index("forge_budget")]
        > supply[RESOURCE_NAMES.index("forge_budget")]
    )


def test_stable_env_xiranite_recipe_needs_less_carbon_than_plain():
    """The Stable-ENV-gated Xiranite recipe (1 Carbon) is confirmed
    cheaper than the plain one (2 Stabilized Carbon) -- this is why
    search() prefers it whenever Stable ENV capacity is available."""
    formulas = build_formulas(WulingConfig1p4())
    plain = formulas["xi_forge_run"]
    stable = formulas["xi_forge_stable_env_run"]
    plain_ratio = (
        plain.consumption[RESOURCE_NAMES.index("stabilized_carbon")]
        / -plain.consumption[RESOURCE_NAMES.index("xi")]
    )
    stable_ratio = (
        stable.consumption[RESOURCE_NAMES.index("carbon")]
        / -stable.consumption[RESOURCE_NAMES.index("xi")]
    )
    assert plain_ratio == pytest.approx(2.0)
    assert stable_ratio == pytest.approx(1.0)


def test_search_prefers_stable_env_xiranite_when_available():
    """With the default config, Stable ENV allowance is a genuinely
    scarce resource shared with other gated recipes (the Purification
    Unit Stable-ENV variants, now economically attractive too since
    Heavy Xiragen/Pyrrolite Gas feed real value), so some Xiranite may
    still go through the plain (more expensive) route once Stable ENV
    capacity runs out. Isolate the Forge-of-the-Sky-specific preference
    by banning the other Stable-ENV-gated recipes, confirming the
    cheaper route is used exclusively once it isn't contested."""
    config = WulingConfig1p4(
        formula_limits={
            "purification_heavy_xiragen_stable_alloc": 0.0,
            "purification_hetonite_gas_stable_alloc": 0.0,
        }
    )
    result, names = search(config)
    rates = dict(zip(names, result.formula_rates))
    assert rates["xi_forge_stable_env_alloc"] > 0.0
    assert rates["xi_forge_alloc"] == pytest.approx(0.0)


# ---- Crafting Point chain (flexible_gear_crafting.md), validated by
# hand against kaneko_1p4_data_sheet.md's real per-tier gear costs: T1
# gears cost 50 XC / 50 CC / 10 HC / 5 PC; T2 costs 50 CC / 10 HC / 5 PC;
# T3 costs 50 HC / 25 PC; T4 costs 50 PC. These tests confirm the
# conversion chain reproduces every one of those numbers exactly. ----


def test_crafting_point_chain_reproduces_real_t1_gear_costs():
    formulas = build_formulas(WulingConfig1p4())
    # 50 Xiranite Component -> 1 T1 Crafting Point (direct)
    assert (
        formulas["component_to_t1"].consumption[
            RESOURCE_NAMES.index("xiranite_component_item")
        ]
        == 50.0
    )
    # 50 Cuprium Component -> 1 T2 Crafting Point -> 1 T1 Crafting Point
    # (T2->T1 ratio 1:1, matching "50 CC also covers 1 T1 gear")
    t2_to_t1 = formulas["t2_to_t1"]
    assert (
        t2_to_t1.consumption[RESOURCE_NAMES.index("t2_crafting_point")]
        / -t2_to_t1.consumption[RESOURCE_NAMES.index("t1_crafting_point")]
        == 1.0
    )
    # 10 Hetonite Component -> 0.2 T3 Crafting Point -> (T3->T2 ratio 5)
    # -> 1.0 T2 Crafting Point -> (T2->T1 ratio 1) -> 1.0 T1 Crafting
    # Point, matching "10 HC also covers 1 T1 gear"
    t3_to_t2 = formulas["t3_to_t2"]
    hc_per_t3 = formulas["component_to_t3"].consumption[
        RESOURCE_NAMES.index("hetonite_component_item")
    ]
    t3_per_t2 = (
        -t3_to_t2.consumption[RESOURCE_NAMES.index("t2_crafting_point")]
        / (t3_to_t2.consumption[RESOURCE_NAMES.index("t3_crafting_point")])
    )
    hc_for_one_t1 = hc_per_t3 / t3_per_t2  # since t2->t1 is 1:1
    assert hc_for_one_t1 == pytest.approx(10.0)


def test_crafting_point_chain_reproduces_real_pyrrolite_component_costs():
    """5 PC covers T1 or T2; 25 PC covers T3; 50 PC covers T4 -- this is
    the specific numeric validation that confirmed the whole Crafting
    Point design (see tmp_notes/wip_todo.md)."""
    formulas = build_formulas(WulingConfig1p4())
    pc_per_t4 = formulas["component_to_t4"].consumption[
        RESOURCE_NAMES.index("pyrrolite_component")
    ]
    t4_to_t3 = formulas["t4_to_t3"]
    t3_to_t2 = formulas["t3_to_t2"]
    t2_to_t1 = formulas["t2_to_t1"]

    t4_per_t3 = (
        -t4_to_t3.consumption[RESOURCE_NAMES.index("t3_crafting_point")]
        / (t4_to_t3.consumption[RESOURCE_NAMES.index("t4_crafting_point")])
    )
    t3_per_t2 = (
        -t3_to_t2.consumption[RESOURCE_NAMES.index("t2_crafting_point")]
        / (t3_to_t2.consumption[RESOURCE_NAMES.index("t3_crafting_point")])
    )
    t2_per_t1 = (
        -t2_to_t1.consumption[RESOURCE_NAMES.index("t1_crafting_point")]
        / (t2_to_t1.consumption[RESOURCE_NAMES.index("t2_crafting_point")])
    )

    pc_for_one_t4 = pc_per_t4
    pc_for_one_t3 = pc_per_t4 / t4_per_t3
    pc_for_one_t2 = pc_for_one_t3 / t3_per_t2
    pc_for_one_t1 = pc_for_one_t2 / t2_per_t1

    assert pc_for_one_t4 == pytest.approx(50.0)
    assert pc_for_one_t3 == pytest.approx(25.0)
    assert pc_for_one_t2 == pytest.approx(5.0)
    assert pc_for_one_t1 == pytest.approx(5.0)


def test_formula_limits_and_outputs_overrides_work():
    config = WulingConfig1p4(formula_limits={"pyrrolite_part_sell": 0.0})
    formulas = build_formulas(config)
    assert formulas["pyrrolite_part_sell"].limit == 0.0


def test_unknown_formula_override_raises():
    with pytest.raises(ValueError):
        build_formulas(WulingConfig1p4(formula_limits={"not_a_real_formula": 1.0}))


# ---- "No folded formulas" (confirmed with the user): 1.2e's own
# yazhen_solution_make/jincao_solution_make collapse Planting+Shredding+
# Reactor Crucible into one zero-cost step; this module unfolds them
# into their real 3 stages, verified lossless against 1.2e's historical
# $-optimal figures exactly. ----


def _1p2e_matching_base_supply():
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ori")] = 540.0
    supply[RESOURCE_NAMES.index("ferr")] = 90.0
    supply[RESOURCE_NAMES.index("cup_ore")] = 240.0
    return supply


# Formulas that replace a 1.2e abstraction with equivalent real-recipe
# capability (Forge of the Sky's new 3-way split, the Carbon chain
# feeding it, and Yazhen/Jincao's unfolded 3 stages) -- required for an
# exact 1.2e reproduction, since without them Xiranite/Yazhen Solution/
# Jincao Solution would have literally no source at all, not just a
# less-generous one. Everything else new_1p4_formula_names finds is
# purely additive (the wider gas economy, Crafting Points, Pyrrolite,
# the two new sell formulas) and gets banned for the exact-match tests
# below -- confirmed empirically, not just reasoned about, that banning
# exactly this complement reproduces the historical figures bit-exactly
# (gases didn't exist in 1.2e at all, and every path to touch them goes
# through a banned formula, so this fully removes them from the model).
_REQUIRED_1P4_INFRASTRUCTURE = frozenset(
    {
        "xi_forge_alloc",
        "xi_forge_stable_env_alloc",
        "xi_forge_run",
        "xi_forge_stable_env_run",
        "hx_forge_alloc",
        "buckflower_plant",
        "carbon_from_buckflower",
        "carbon_from_sandleaf",
        "carbon_from_jincao",
        "carbon_from_yazhen",
        "carbon_powder_make",
        "dense_carbon_powder_make",
        "stabilized_carbon_make",
        "yazhen_plant",
        "yazhen_powder_make",
        "jincao_plant",
        "jincao_powder_make",
    }
)


def _1p2e_equivalent_formula_limits(config: WulingConfig1p4) -> dict[str, float]:
    """Ban every purely-additive 1.4 formula (the wider gas economy,
    Crafting Points, Pyrrolite, the two new sell formulas), keeping only
    the required replacement infrastructure active -- see
    _REQUIRED_1P4_INFRASTRUCTURE."""
    to_ban = _new_1p4_formula_names(config) - _REQUIRED_1P4_INFRASTRUCTURE
    return {name: 0.0 for name in to_ban}


def test_reproduces_1p2e_historical_dollar_figure_exactly_with_gas_economy_disabled():
    """The core "no folded formulas, should match past results" check,
    done properly per the user's own suggestion: rather than relying on
    zero Inergen/Xiragen base supply alone (insufficient -- gases can
    still be bootstrapped from otherwise-abundant Xiranite/Liquid
    Xiranite via the Fluid-Gas/Solid-Gas Transmuting Unit's own
    self-referential recipes, which is a real, intentional capability,
    not a bug), explicitly ban every formula that isn't required
    replacement infrastructure for Forge of the Sky/the Carbon chain/
    Yazhen-Jincao's unfolding. With the wider gas economy fully and
    explicitly disabled this way, the fully-unfolded model reproduces
    1.2e's historical $1415.99... (206735/146) figure exactly, not just
    "at least" -- confirming the unfolding itself is truly lossless."""
    probe_config = WulingConfig1p4(base_supply=_1p2e_matching_base_supply())
    config = WulingConfig1p4(
        base_supply=_1p2e_matching_base_supply(),
        formula_limits=_1p2e_equivalent_formula_limits(probe_config),
    )
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output == pytest.approx(206735 / 146)


def test_reproduces_1p2e_ban_ya_historical_figure_exactly_with_gas_economy_disabled():
    """Same as above, but the ban-ya historical invariant."""
    probe_config = WulingConfig1p4(base_supply=_1p2e_matching_base_supply())
    config = WulingConfig1p4(
        base_supply=_1p2e_matching_base_supply(),
        formula_limits={
            **_1p2e_equivalent_formula_limits(probe_config),
            "ya": 0.0,
            "jincao_tea": 0.0,
        },
    )
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output == pytest.approx(205129 / 146)


def test_matches_or_exceeds_1p2e_historical_dollar_figure():
    """The core "no folded formulas, should match past results" check:
    given the same base supply 1.2e's own historical $-optimal test uses
    (ori=540, ferr=90, cup_ore=240, no Inergen/Xiragen), wuling_1p4's
    fully-unfolded model must reach AT LEAST the historical $1415.99...
    (206735/146) figure -- not exactly, anymore, now that the Fluid-Gas/
    Solid-Gas Transmuting Unit's reverse recipes are modeled too (see
    test_pyrrolite_has_no_base_supply_but_is_reachable_via_reverse_recipe):
    those genuinely unlock new value 1.2e never had access to (e.g.
    bootstrapping Xiragen from otherwise-"free" Xiranite via
    solid_gas_xiragen, then routing it through the wider gas network),
    even starting from zero Inergen/Xiragen base supply. This is
    expected, not a regression -- adding more feasible options to an
    LP's search space can only weakly improve its optimum, never hurt
    it (basic LP monotonicity), so ">=" is the correct invariant here,
    not "==". The unfolding's own correctness (not just "doesn't regress
    the $ figure") is separately verified at the ratio level by
    test_yazhen_and_jincao_are_fully_unfolded_not_collapsed,
    test_yazhen_jincao_unfolded_ratio_preserves_1p2e_batch_size, and the
    Carbon-chain ratio tests below, which don't depend on what the wider
    gas network does.

    This caught two real bugs before even reaching that "no regression"
    bar: (1) WulingConfig1p4 was passing metatransfers=[] to the
    underlying 1.2e config, silently making metatransfer_option_0 --
    used at rate 1.0 in the real historical solution -- permanently
    unavailable; (2) full_supply() never credited metatransfer_allowance
    at all, so enabling metatransfers alone wouldn't have been enough
    either. Also surfaced that 1.2e's own sandleaf_plant limit (5) is
    too tight once Sandleaf gets a new competing consumer (the Carbon
    chain) -- see DEFAULT_PLANTING_LIMIT's docstring."""
    config = WulingConfig1p4(base_supply=_1p2e_matching_base_supply())
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output >= 206735 / 146 - 1e-6


def test_matches_or_exceeds_1p2e_ban_ya_historical_figure():
    """Second historical invariant, banning ya/jincao_tea (see
    wuling.py's own test for why both must be banned together --
    they're perfect economic substitutes). See
    test_matches_or_exceeds_1p2e_historical_dollar_figure for why ">="
    rather than "==" is the correct invariant now."""
    config = WulingConfig1p4(
        base_supply=_1p2e_matching_base_supply(),
        formula_limits={"ya": 0.0, "jincao_tea": 0.0},
    )
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output >= 205129 / 146 - 1e-6


def test_metatransfer_option_available_and_credited():
    """Regression: WulingConfig1p4 used to pass metatransfers=[] to the
    underlying 1.2e config, and full_supply() never credited
    metatransfer_allowance either -- either bug alone silently made
    metatransfer_option_0 permanently unusable."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    assert "metatransfer_option_0" in formulas
    supply = full_supply(config)
    assert supply[RESOURCE_NAMES.index("metatransfer_allowance")] >= 1.0


def test_yazhen_and_jincao_are_fully_unfolded_not_collapsed():
    """The old 1.2e formulas (yazhen_solution_make: {"yazhen_solution":
    -30}, entirely free) must be gone, replaced by the real 3-stage
    chain -- Planting (yazhen_plant), Shredding (yazhen_powder_make),
    Reactor Crucible (yazhen_solution_make, redefined) -- each a
    genuine Formula with the recipe's own resource, not a single
    zero-cost step."""
    formulas = build_formulas(WulingConfig1p4())
    for prefix, raw, powder, solution in (
        ("yazhen", "yazhen_raw", "yazhen_powder", "yazhen_solution"),
        ("jincao", "jincao_raw", "jincao_powder", "jincao_solution"),
    ):
        plant = formulas[f"{prefix}_plant"]
        powder_make = formulas[f"{prefix}_powder_make"]
        solution_make = formulas[f"{prefix}_solution_make"]
        assert plant.consumption[RESOURCE_NAMES.index(raw)] < 0
        assert powder_make.consumption[RESOURCE_NAMES.index(raw)] > 0
        assert powder_make.consumption[RESOURCE_NAMES.index(powder)] < 0
        assert solution_make.consumption[RESOURCE_NAMES.index(powder)] > 0
        assert solution_make.consumption[RESOURCE_NAMES.index(solution)] < 0
        # the old folded formula produced Solution directly from
        # nothing -- the redefined one must consume real Powder instead
        assert not all(
            amount == 0 or resource_name == solution
            for resource_name, amount in zip(RESOURCE_NAMES, solution_make.consumption)
        )


def test_yazhen_jincao_unfolded_ratio_preserves_1p2e_batch_size():
    """1.2e's folded yazhen_solution_make produced exactly 30 Yazhen
    Solution per multiple; the unfolded 3-stage chain must still need
    exactly 15 raw Yazhen to reach that same 30 Yazhen Solution (30
    Yazhen Solution -> 30 Yazhen Powder [Reactor Crucible 1:1] -> 15
    Yazhen [Shredding 1:2]), so this is a lossless reformulation, not
    just a differently-shaped one."""
    formulas = build_formulas(WulingConfig1p4())
    for prefix, raw, powder, solution in (
        ("yazhen", "yazhen_raw", "yazhen_powder", "yazhen_solution"),
        ("jincao", "jincao_raw", "jincao_powder", "jincao_solution"),
    ):
        powder_make = formulas[f"{prefix}_powder_make"]
        solution_make = formulas[f"{prefix}_solution_make"]
        # multiples of solution_make needed for 30 Solution:
        solution_multiples = (
            30.0 / -solution_make.consumption[RESOURCE_NAMES.index(solution)]
        )
        powder_needed = (
            solution_multiples * solution_make.consumption[RESOURCE_NAMES.index(powder)]
        )
        powder_multiples = (
            powder_needed / -powder_make.consumption[RESOURCE_NAMES.index(powder)]
        )
        raw_needed = (
            powder_multiples * powder_make.consumption[RESOURCE_NAMES.index(raw)]
        )
        assert raw_needed == pytest.approx(15.0)


def test_carbon_chain_has_all_four_confirmed_sources():
    """Buckflower and Sandleaf (30:30) and Jincao and Yazhen (30:60,
    twice as efficient) must all be usable, competing Carbon sources --
    per tmp_notes/old_prompt.md's "Refining Unit: 30 Buckflower OR 30
    Sandleaf -> 30 Carbon" / "30 Jincao OR 30 Yazhen -> 60 Carbon"."""
    formulas = build_formulas(WulingConfig1p4())
    for formula_name, input_name, ratio in (
        ("carbon_from_buckflower", "buckflower", 1.0),
        ("carbon_from_sandleaf", "sandleaf_raw", 1.0),
        ("carbon_from_jincao", "jincao_raw", 0.5),
        ("carbon_from_yazhen", "yazhen_raw", 0.5),
    ):
        formula = formulas[formula_name]
        input_amount = formula.consumption[RESOURCE_NAMES.index(input_name)]
        carbon_amount = -formula.consumption[RESOURCE_NAMES.index("carbon")]
        assert input_amount / carbon_amount == pytest.approx(ratio)


def test_stabilized_carbon_chain_ratios_match_old_prompt():
    """Carbon -> Carbon Powder (1:2), + Sandleaf Powder -> Dense Carbon
    Powder (2:1, plus a real Sandleaf Powder co-input), -> Stabilized
    Carbon (1:1) -- per tmp_notes/old_prompt.md."""
    formulas = build_formulas(WulingConfig1p4())
    carbon_powder_make = formulas["carbon_powder_make"]
    assert carbon_powder_make.consumption[
        RESOURCE_NAMES.index("carbon")
    ] / -carbon_powder_make.consumption[
        RESOURCE_NAMES.index("carbon_powder")
    ] == pytest.approx(0.5)

    dense = formulas["dense_carbon_powder_make"]
    assert dense.consumption[
        RESOURCE_NAMES.index("carbon_powder")
    ] / -dense.consumption[
        RESOURCE_NAMES.index("dense_carbon_powder")
    ] == pytest.approx(2.0)
    assert dense.consumption[RESOURCE_NAMES.index("sandleaf")] > 0.0

    stabilized = formulas["stabilized_carbon_make"]
    assert stabilized.consumption[
        RESOURCE_NAMES.index("dense_carbon_powder")
    ] / -stabilized.consumption[
        RESOURCE_NAMES.index("stabilized_carbon")
    ] == pytest.approx(1.0)


# ---- Virtual power/Water/Acid $ tax (WulingConfig1p4.power_dollar_tax) ----


def test_power_dollar_tax_resolves_forge_split_tie_deterministically():
    """Without the tax (see test_power_dollar_tax_disabled_restores_old_
    tie below), the default scenario's Forge of the Sky Carbon-sourcing
    split is a genuine LP tie (many splits summing to 12 forges are
    equally optimal). With the tax (the default), the LP has a real,
    non-arbitrary reason to prefer the all-Stable-ENV route (needs half
    the Carbon per Xiranite) -- confirmed empirically this is now
    deterministic: every forge goes to the Stable ENV route, none to
    the plain (Stabilized Carbon) route."""
    config = WulingConfig1p4()
    result, names = search(config)
    rates = dict(zip(names, result.formula_rates))
    assert rates.get("xi_forge_stable_env_alloc", 0.0) == pytest.approx(12.0)
    assert rates.get("xi_forge_alloc", 0.0) == pytest.approx(0.0)


def test_power_dollar_tax_disabled_restores_old_tie():
    """power_dollar_tax=False reproduces the pre-tax behavior exactly:
    the Forge of the Sky split is no longer forced to the all-Stable-ENV
    route -- confirmed via HiGHS's own (arbitrary but deterministic for
    a fixed problem) tie-breaking choosing a mixed 3/9 split instead,
    same as before the tax was ever introduced.

    The DEFAULT scenario's own Inergen/Xiragen supply is now high enough
    (460/100, up from the original 260/30 -- see DEFAULT_INERGEN/
    DEFAULT_XIRAGEN's own comments) that the all-Stable-ENV route wins
    outright even with the tax off, for reasons unrelated to the tax
    (plain abundance of gas-economy inputs) -- so reproducing the
    original tie needs the original, smaller Inergen/Xiragen supply
    explicitly, isolating the tax's own effect from that separate,
    supply-driven change."""
    supply = _default_base_supply()
    supply[RESOURCE_NAMES.index("inergen")] = 260.0
    supply[RESOURCE_NAMES.index("xiragen")] = 30.0
    config = WulingConfig1p4(power_dollar_tax=False, base_supply=supply)
    result, names = search(config)
    rates = dict(zip(names, result.formula_rates))
    assert rates.get("xi_forge_alloc", 0.0) > 0.0
    assert rates.get("xi_forge_stable_env_alloc", 0.0) > 0.0


def test_power_dollar_tax_never_changes_the_true_optimal_dollar_value():
    """The tax only ever guides WHICH tied vertex the LP picks -- it must
    never change the reported dollar_output, since search() backs the
    tax back out via power_dollar_tax_paid() before returning (confirmed
    with the user: the tax's own $ amount has no real-world meaning, it
    deliberately covers only some buildings). Both the tax-guided and
    untaxed searches must report the exact same historically-tied $
    figure."""
    with_tax, _ = search(WulingConfig1p4(power_dollar_tax=True))
    without_tax, _ = search(WulingConfig1p4(power_dollar_tax=False))
    assert with_tax.dollar_output == pytest.approx(without_tax.dollar_output, abs=1e-6)


def test_power_dollar_tax_paid_matches_manual_computation():
    """power_dollar_tax_paid() must exactly equal sum(rate * watts *
    $/W) over every FORMULA_WATTS entry, computed independently here."""
    dollar_per_watt = 54.0 / (3200.0 / 1.5)
    rates_by_name = {name: float(i + 1) for i, name in enumerate(FORMULA_WATTS)}
    rates_by_name["some_unrelated_formula"] = 999.0  # must be ignored
    expected = sum(
        rate * FORMULA_WATTS[name] * dollar_per_watt
        for name, rate in rates_by_name.items()
        if name in FORMULA_WATTS
    )
    assert power_dollar_tax_paid(rates_by_name) == pytest.approx(expected)


def test_power_dollar_tax_paid_zero_for_no_rates():
    assert power_dollar_tax_paid({}) == 0.0


def test_formula_watts_formulas_get_negative_output_when_tax_enabled():
    """Every FORMULA_WATTS-taxed formula (resolving the {name}_run
    variant for threshold recipes, since the formula-name set no longer
    depends on continuous_thresholds -- see _threshold_formulas) must
    have a strictly negative .output when the tax is on, and exactly
    0.0 when it's off."""
    taxed = build_formulas(WulingConfig1p4(power_dollar_tax=True))
    untaxed = build_formulas(WulingConfig1p4(power_dollar_tax=False))
    for name in FORMULA_WATTS:
        actual_name = name if name in taxed else f"{name}_run"
        assert taxed[actual_name].output < 0.0
        assert untaxed[actual_name].output == 0.0


# ---- _threshold_formulas: formula-name set is invariant to
# continuous_thresholds (only .integer changes) ----


def test_continuous_thresholds_does_not_change_formula_names():
    integer_names = set(build_formulas(WulingConfig1p4(continuous_thresholds=False)))
    continuous_names = set(build_formulas(WulingConfig1p4(continuous_thresholds=True)))
    assert integer_names == continuous_names


def test_continuous_thresholds_toggles_alloc_integer_flag_only():
    integer_formulas = build_formulas(WulingConfig1p4(continuous_thresholds=False))
    continuous_formulas = build_formulas(WulingConfig1p4(continuous_thresholds=True))
    for name, *_ in _THRESHOLD_RECIPES:
        assert integer_formulas[f"{name}_alloc"].integer is True
        assert continuous_formulas[f"{name}_alloc"].integer is False
        # _run's own shape (consumption/output) must be identical either
        # way -- only the alloc's integrality changes.
        assert np.array_equal(
            integer_formulas[f"{name}_run"].consumption,
            continuous_formulas[f"{name}_run"].consumption,
        )


def test_threshold_recipes_reverse_derivation_matches_hand_verified_values():
    """Regression for _reverse_threshold_recipe (auto-derives all 11
    reverse recipes from the 11 forward ones via sign-flip): spot-check
    a plain recipe, a batch (per-cycle) recipe, and both self-referential
    recipes against the exact values hand-verified against
    kaneko_1p4_data_sheet.md/old_prompt.md in the original (pre-refactor)
    hand-written table."""
    recipes = {
        name: (threshold_good, max_rate, other)
        for name, threshold_good, max_rate, other in _THRESHOLD_RECIPES
    }

    assert recipes["fluid_gas_aquagen_reverse"] == (
        "liquid_xiranite",
        30.0,
        {"aquagen": 1.0},
    )
    assert recipes["fluid_gas_heavy_xiragen_reverse"] == (
        "liquid_xiranite",
        6.0,
        {"heavy_xiragen": 5.0, "liquid_heavy_xiranite": -2.0},
    )
    # Self-referential: the threshold good is ALSO the recipe's own
    # reactant/product, so the reverse's sign flip must apply to that
    # SAME combined entry too, not just the "other" item.
    assert recipes["fluid_gas_xiragen_reverse"] == (
        "liquid_xiranite",
        30.0,
        {"liquid_xiranite": -1.0, "xiragen": 1.0},
    )
    assert recipes["solid_gas_xiragen_reverse"] == (
        "xiragen",
        30.0,
        {"xiragen": 1.0, "xi": -1.0},
    )


# ---- "1 multiple = 1 building's real /min rate" scaling, not a raw
# per-cycle item count (user-reported: "Forge of the Sky, Stable ENV:
# 1/min Carbon + 1/min Water -> 1/min Xiranite" was wrong -- every
# "every 2 seconds"/"every 10 seconds" recipe must convert to 30/min or
# 6/min as its base per-multiple rate) ----


def test_threshold_run_formulas_scale_other_consumption_by_max_rate():
    """_threshold_formulas' run_vec must be other_consumption scaled by
    max_rate (30 or 6), not the raw per-cycle magnitude -- confirmed
    against fluid_gas_aquagen (max_rate=30, {"aquagen": -1.0} ->
    -30.0/min) and fluid_gas_heavy_xiragen (max_rate=6, batch recipe,
    {"liquid_heavy_xiranite": 2.0, "heavy_xiragen": -5.0} -> 12.0/-30.0)."""
    formulas = build_formulas(WulingConfig1p4())
    aquagen_run = formulas["fluid_gas_aquagen_run"]
    assert aquagen_run.consumption[RESOURCE_NAMES.index("aquagen")] == -30.0
    heavy_xiragen_run = formulas["fluid_gas_heavy_xiragen_run"]
    assert (
        heavy_xiragen_run.consumption[RESOURCE_NAMES.index("liquid_heavy_xiranite")]
        == 12.0
    )
    assert heavy_xiragen_run.consumption[RESOURCE_NAMES.index("heavy_xiragen")] == -30.0


def test_threshold_alloc_mints_exactly_one_capacity_unit_per_building():
    """Every {name}_alloc must mint exactly 1 unit of {name}_capacity per
    building (not max_rate units) -- matching 1.2e's own hx_forge_alloc/
    hx_make precedent, so 1 run-multiple = 1 fully-committed building."""
    formulas = build_formulas(WulingConfig1p4())
    for name, *_ in _THRESHOLD_RECIPES:
        capacity_idx = RESOURCE_NAMES.index(f"{name}_capacity")
        assert formulas[f"{name}_alloc"].consumption[capacity_idx] == -1.0
        assert formulas[f"{name}_run"].consumption[capacity_idx] == 1.0


def test_xi_forge_run_formulas_produce_30_per_min_xiranite_per_multiple():
    """Forge of the Sky's real recipe ("1 Carbon + 1 Water -> 1 Xiranite
    every 2 seconds") saturates at 30/min per building -- xi_forge_run/
    xi_forge_stable_env_run must reflect that, not the raw 1-per-cycle
    count (the exact bug reported: the CLI showed "1/min Carbon + 1/min
    Water -> 1/min Xiranite" instead of the real 30/min)."""
    formulas = build_formulas(WulingConfig1p4())
    assert formulas["xi_forge_run"].consumption[RESOURCE_NAMES.index("xi")] == -30.0
    assert (
        formulas["xi_forge_run"].consumption[RESOURCE_NAMES.index("stabilized_carbon")]
        == 60.0
    )
    assert (
        formulas["xi_forge_stable_env_run"].consumption[RESOURCE_NAMES.index("xi")]
        == -30.0
    )
    assert (
        formulas["xi_forge_stable_env_run"].consumption[RESOURCE_NAMES.index("carbon")]
        == 30.0
    )
    assert (
        formulas["xi_forge_alloc"].consumption[
            RESOURCE_NAMES.index("xi_forge_capacity")
        ]
        == -1.0
    )


def test_gearing_unit_matches_the_other_gear_components_own_scale():
    """gearing_unit (Pyrrolite Component's own 1.4 producer) must use the
    SAME "1 multiple = 1 building = 6 Component/min" scale the other four
    1.2e-legacy Gear Component formulas already use (e.g. hetonite_component:
    12 Hetonite Part + 12 Heavy Xiranite -> 6 Hetonite Component) -- an
    earlier version left it at the raw "1 Pyrrolite -> 1 Pyrrolite
    Component" per-cycle count instead."""
    formulas = build_formulas(WulingConfig1p4())
    gearing_unit = formulas["gearing_unit"]
    assert gearing_unit.consumption[RESOURCE_NAMES.index("pyrrolite")] == 6.0
    assert gearing_unit.consumption[RESOURCE_NAMES.index("heavy_xiranite")] == 12.0
    assert gearing_unit.consumption[RESOURCE_NAMES.index("pyrrolite_component")] == -6.0
    assert GOOD_YIELD["gearing_unit"] == 6.0


def test_scaling_fix_does_not_change_the_true_optimal_dollar_value():
    """This whole fix is a pure change of variables (scale run-rate
    variables down by max_rate, scale their coefficients up to match) --
    the $-optimal value of the default scenario must be bit-for-bit
    unchanged by it. Regression pin against the current default figure,
    so a future accidental re-introduction of a stray un-rescaled
    coefficient (breaking the change-of-variables equivalence) would
    show up as a $ change here."""
    result, _ = search(WulingConfig1p4())
    assert result.dollar_output == pytest.approx(606727 / 250)
