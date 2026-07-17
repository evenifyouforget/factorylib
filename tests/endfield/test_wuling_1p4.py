import numpy as np
import pytest

from factorylib.endfield.wuling_1p4 import (
    RESOURCE_NAMES,
    WulingConfig1p4,
    build_formulas,
    full_supply,
    search,
)


def test_build_formulas_does_not_raise():
    formulas = build_formulas(WulingConfig1p4())
    assert len(formulas) > 0


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
