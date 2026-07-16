import numpy as np

from factorylib.endfield.goals import (
    ProductionPlan,
    _plan_complexity,
    item_rates,
    plan_from_search_result,
)
from factorylib.endfield.wuling import WulingConfig, search


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


def test_forge_and_metatransfer_bookkeeping_formulas_are_complexity_free():
    """xiranite_forge_alloc/heavy_xiranite_forge_alloc/metatransfer_
    option_* are pure bookkeeping (see wuling.is_forge_or_metatransfer_
    formula) -- a player never builds a belt for them, so their own
    (often ugly, LP-vertex-driven) fraction shouldn't count against a
    plan's complexity."""
    plan = ProductionPlan(dollar_rate=0.0, multiples={"xiranite_forge_alloc": 19 / 96})
    assert _plan_complexity(plan, 1000) == 0.0


def test_pp_bookkeeping_formulas_are_complexity_free():
    """Same reasoning for pp-tier/bonus/delivery-quota formulas (see
    pp_goals.is_pp_bookkeeping_formula) -- delivery_quota_from_sandleaf
    genuinely touches a real resource (sandleaf), but pricing its OWN
    fraction separately would double-count complexity already priced on
    the real formula that actually produces that resource."""
    plan = ProductionPlan(
        dollar_rate=0.0,
        multiples={"delivery_quota_from_sandleaf": 19 / 96, "pp_dollar_1": 13 / 17},
    )
    assert _plan_complexity(plan, 1000) == 0.0


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
