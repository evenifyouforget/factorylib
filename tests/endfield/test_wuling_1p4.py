import numpy as np
import pytest

from factorylib.endfield.wuling_1p4 import (
    _NEW_RESOURCE_NAMES,
    RESOURCE_NAMES,
    WulingConfig1p4,
    build_formulas,
    full_supply,
    search,
)


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

    "pyrrolite" is deliberately excluded: confirmed with the user it
    must have no base supply (it's crafted, not sourced directly), but
    no confirmed recipe produces it either -- a known, flagged gap (see
    _default_base_supply's docstring), not an accidental one like
    liquid_heavy_xiranite was."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    supply = full_supply(config)
    checked_names = [name for name in _NEW_RESOURCE_NAMES if name != "pyrrolite"]
    producers: dict[str, list[str]] = {name: [] for name in checked_names}
    for formula_name, formula in formulas.items():
        for resource_name in checked_names:
            idx = RESOURCE_NAMES.index(resource_name)
            if formula.consumption[idx] < 0:
                producers[resource_name].append(formula_name)

    unreachable = [
        name
        for name in checked_names
        if not producers[name] and supply[RESOURCE_NAMES.index(name)] <= 0.0
    ]
    assert unreachable == []


def test_pyrrolite_has_no_base_supply_and_is_currently_unreachable():
    """Documents the known gap (see _default_base_supply's docstring):
    Pyrrolite must have zero base supply (confirmed with the user), and
    no confirmed recipe produces it, so it's currently completely
    unreachable in this model -- not a bug, but should stay visible so
    it isn't silently "fixed" by a guessed reverse-recipe rate later
    without deliberately deciding to do so."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    supply = full_supply(config)
    assert supply[RESOURCE_NAMES.index("pyrrolite")] == 0.0
    assert not any(
        formula.consumption[RESOURCE_NAMES.index("pyrrolite")] < 0
        for formula in formulas.values()
    )


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
    result, names = search(WulingConfig1p4())
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


def test_reproduces_1p2e_historical_dollar_figure_exactly():
    """The core "no folded formulas, should match past results" check:
    given the same base supply 1.2e's own historical $-optimal test uses
    (ori=540, ferr=90, cup_ore=240, no Inergen/Xiragen), wuling_1p4's
    fully-unfolded model must reproduce the exact same $1415.99...
    (206735/146) figure -- not approximately, exactly, since every new
    formula here is either a lossless pass-through or genuinely unused
    at this base supply level.

    This caught two real bugs before passing: (1) WulingConfig1p4 was
    passing metatransfers=[] to the underlying 1.2e config, silently
    making metatransfer_option_0 -- used at rate 1.0 in the real
    historical solution -- permanently unavailable; (2) full_supply()
    never credited metatransfer_allowance at all, so enabling
    metatransfers alone wouldn't have been enough either. Also
    surfaced that 1.2e's own sandleaf_plant limit (5) is too tight
    once Sandleaf gets a new competing consumer (the Carbon chain) --
    see DEFAULT_PLANTING_LIMIT's docstring."""
    config = WulingConfig1p4(base_supply=_1p2e_matching_base_supply())
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output == pytest.approx(206735 / 146)


def test_reproduces_1p2e_ban_ya_historical_figure():
    """Second historical invariant, banning ya/jincao_tea (see
    wuling.py's own test for why both must be banned together --
    they're perfect economic substitutes)."""
    config = WulingConfig1p4(
        base_supply=_1p2e_matching_base_supply(),
        formula_limits={"ya": 0.0, "jincao_tea": 0.0},
    )
    result, names = search(config)
    assert result.status == "optimal"
    assert result.dollar_output == pytest.approx(205129 / 146)


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
