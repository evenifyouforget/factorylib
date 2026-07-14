import numpy as np

from factorylib.endfield.goals import (
    ProductionPlan,
    WulingGoals,
    _plan_complexity,
    fitness,
    item_rates,
    plan_from_search_result,
)
from factorylib.endfield.wuling import WulingConfig, search


def _plan(dollar_rate=0.0, power_rate=0.0, good_rates=None, multiples=None):
    return ProductionPlan(
        dollar_rate=dollar_rate,
        power_rate=power_rate,
        good_rates=good_rates or {},
        multiples=multiples or {},
    )


def test_below_cap_scores_lower_than_at_cap():
    goals = WulingGoals()
    below = fitness(_plan(dollar_rate=goals.stock_bill_cap * 0.5), goals)
    at_cap = fitness(_plan(dollar_rate=goals.stock_bill_cap), goals)
    assert below < at_cap


def test_reaching_target_ratio_scores_higher_than_at_cap():
    goals = WulingGoals()
    at_cap = fitness(_plan(dollar_rate=goals.stock_bill_cap), goals)
    at_target = fitness(
        _plan(dollar_rate=goals.stock_bill_cap * goals.stock_bill_target_ratio), goals
    )
    assert at_target > at_cap


def test_excess_past_target_ratio_has_diminishing_returns():
    goals = WulingGoals()
    at_target = goals.stock_bill_cap * goals.stock_bill_target_ratio
    step = goals.stock_bill_cap  # equal-sized absolute steps
    f_at_target = fitness(_plan(dollar_rate=at_target), goals)
    f_plus_1 = fitness(_plan(dollar_rate=at_target + step), goals)
    f_plus_2 = fitness(_plan(dollar_rate=at_target + 2 * step), goals)
    # more is never worse ...
    assert f_plus_1 > f_at_target
    assert f_plus_2 > f_plus_1
    # ... but each successive equal-sized step forward gains less.
    gain_1 = f_plus_1 - f_at_target
    gain_2 = f_plus_2 - f_plus_1
    assert gain_2 < gain_1


def test_more_is_never_penalized():
    goals = WulingGoals()
    lo = fitness(_plan(dollar_rate=goals.stock_bill_cap * 10), goals)
    hi = fitness(_plan(dollar_rate=goals.stock_bill_cap * 100), goals)
    assert hi >= lo


def test_missing_power_and_gear_goals_score_as_no_requirement():
    goals = WulingGoals()  # power_target=7000 by default, but...
    goals.power_target = 0.0  # simulate "not modeled" -> no term contribution
    with_power_off = fitness(_plan(dollar_rate=goals.stock_bill_cap), goals)
    goals2 = WulingGoals(power_target=0.0)
    same = fitness(_plan(dollar_rate=goals.stock_bill_cap), goals2)
    assert with_power_off == same


def test_empty_delivery_and_gear_goals_contribute_nothing():
    goals = WulingGoals(delivery_goods={}, gear_priority=[])
    plan = _plan(dollar_rate=goals.stock_bill_cap)
    # Score should equal the stock-bill term alone (no complexity, no
    # power target hit in this test either since default power_target
    # applies a shortfall penalty -- zero it out to isolate the claim).
    goals.power_target = 0.0
    assert fitness(plan, goals) == fitness(
        _plan(dollar_rate=goals.stock_bill_cap, good_rates={"anything": 999}), goals
    )


def test_zero_stock_bill_cap_means_no_requirement():
    goals = WulingGoals(stock_bill_cap=0.0)
    assert fitness(_plan(dollar_rate=0.0), goals) == fitness(
        _plan(dollar_rate=1e9), goals
    )


def test_delivery_good_below_target_is_penalized():
    goals = WulingGoals(delivery_goods={"sandleaf_powder": 1000.0})
    low = fitness(_plan(good_rates={"sandleaf_powder": 0.0}), goals)
    high = fitness(_plan(good_rates={"sandleaf_powder": 1000.0}), goals)
    assert high > low


def test_power_below_target_is_penalized():
    goals = WulingGoals()
    low_power = fitness(_plan(power_rate=0.0), goals)
    high_power = fitness(_plan(power_rate=goals.power_target), goals)
    assert high_power > low_power


def test_gear_priority_decay_weights_earlier_ranks_more():
    goals = WulingGoals(
        gear_priority=["parts", "components"],
        gear_min_target=0.5,
        gear_priority_decay=0.5,
    )
    # Satisfying only the top-priority good should score higher than
    # satisfying only the lower-priority one.
    only_top = fitness(_plan(good_rates={"parts": 1.0, "components": 0.0}), goals)
    only_bottom = fitness(_plan(good_rates={"parts": 0.0, "components": 1.0}), goals)
    assert only_top > only_bottom


def test_simpler_multiples_score_higher_than_complex_ones():
    goals = WulingGoals()
    simple = fitness(_plan(multiples={"sc": 0.5}), goals)
    complex_ = fitness(_plan(multiples={"sc": 19 / 96}), goals)
    assert simple > complex_


def test_whole_belt_flow_is_free_even_with_awkward_multiples_fraction():
    """A rate whose "multiples" fraction looks complex (1/4) can still be
    physically trivial if the resource flow it induces lands on exactly
    one whole belt (30 items/min): e.g. a formula consuming 120/run of
    some solid, run at rate 1/4, draws exactly 30/min -- one full belt,
    no splitting at all. This is the concrete case raised in
    factorylib_tmp_physical_factory_construction.md (adding one belt of
    Dense Originium Powder to an SC Wuling Battery-like recipe)."""
    whole_belt = ProductionPlan(
        dollar_rate=0.0,
        multiples={"sc": 0.25},
        consumption={"sc": np.array([0, 120, 0, 0, 0, 0, 0, 0], dtype=float)},
    )
    assert _plan_complexity(whole_belt, 1000) == 0.0


def test_awkward_belt_flow_is_still_priced_even_with_simple_multiples():
    """The converse: the same "1/4 multiple" rate should still be priced
    if the resulting flow does NOT land on a whole belt (here 30/run ->
    7.5/min = 1/4 belt)."""
    awkward = ProductionPlan(
        dollar_rate=0.0,
        multiples={"sc": 0.25},
        consumption={"sc": np.array([0, 30, 0, 0, 0, 0, 0, 0], dtype=float)},
    )
    assert _plan_complexity(awkward, 1000) > 0.0


def test_missing_consumption_falls_back_to_raw_rate_pricing():
    from factorylib.simplicity import fraction_complexity

    plan = ProductionPlan(dollar_rate=0.0, multiples={"sc": 19 / 96})
    assert _plan_complexity(plan, 1000) == fraction_complexity(19 / 96, 1000)


def test_default_gear_priority_and_delivery_goods_are_populated():
    """Now that Components/Sandleaf Powder are modeled (see
    factorylib.endfield.wuling), the defaults should reference them
    rather than stay empty."""
    goals = WulingGoals()
    assert goals.gear_priority == [
        "hetonite_component",
        "xiranite_component",
        "cuprium_component",
        "ferrium_component",
    ]
    assert goals.delivery_goods == {"sandleaf_powder": 15.0}


def test_default_gear_priority_penalizes_missing_components():
    goals = WulingGoals()
    with_components = fitness(
        _plan(
            good_rates={name: 0.5 for name in goals.gear_priority},
        ),
        goals,
    )
    without_components = fitness(_plan(good_rates={}), goals)
    assert with_components > without_components


def test_plan_from_search_result_uses_real_dollar_and_multiples():
    config = WulingConfig()
    result = search(config)
    plan = plan_from_search_result(result, config)
    expected_multiples = dict(zip(result.formula_names, result.result.formula_rates))
    assert plan.dollar_rate == result.result.dollar_output
    assert plan.multiples == expected_multiples
    # good_rates is every formula's rate by name, scaled to real items/min
    # via GOOD_YIELD (now covers the $-earning formulas too, not just the
    # secondary-goal ones).
    assert plan.good_rates == item_rates(expected_multiples)
    # thermal_bank's rate is 0 in the $-maximizing LP optimum (it has no
    # $ value), so power_rate is 0 here too -- not because power isn't
    # modeled, but because the raw LP has no incentive to produce it.
    assert plan.power_rate == 0.0
