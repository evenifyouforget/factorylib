import numpy as np

from factorylib.endfield.goals import ProductionPlan, _plan_complexity
from factorylib.endfield.pp_goals import is_pp_bookkeeping_formula
from factorylib.endfield.pp_goals_1p4 import (
    ALL_NAMES,
    DOLLAR_EARNER_OUTPUTS,
    PPGoals1p4,
    build_pp_formulas,
    pp_supply,
)
from factorylib.endfield.refine import refine_1p4
from factorylib.endfield.wuling_1p4 import (
    RESOURCE_BELT_SPEED,
    RESOURCE_NAMES,
    WulingConfig1p4,
    is_forge_or_metatransfer_formula,
    search,
)
from factorylib.search import SearchConfig


def _is_1p4_bookkeeping(name: str) -> bool:
    return is_forge_or_metatransfer_formula(name) or is_pp_bookkeeping_formula(name)


def test_refine_1p4_never_returns_worse_than_the_initial_baseline():
    config = WulingConfig1p4()
    base_result, base_names = search(config)
    pp_goals = PPGoals1p4()
    baseline_plan = ProductionPlan(
        dollar_rate=0.0,
        multiples=dict(zip(base_names, base_result.formula_rates)),
    )
    baseline_fitness = -pp_goals.complexity_weight * _plan_complexity(
        baseline_plan,
        pp_goals.max_denom,
        resource_names=RESOURCE_NAMES,
        resource_belt_speed=RESOURCE_BELT_SPEED,
        is_bookkeeping_formula=_is_1p4_bookkeeping,
    )

    result = refine_1p4(
        base_result,
        base_names,
        config,
        pp_goals,
        SearchConfig(iterations=3000, seed=42),
        backend="sa",
    )
    assert result.fitness >= baseline_fitness


def test_refine_1p4_sa_result_is_feasible():
    config = WulingConfig1p4()
    base_result, base_names = search(config)
    pp_goals = PPGoals1p4()
    result = refine_1p4(
        base_result, base_names, config, pp_goals, SearchConfig(iterations=1000, seed=1)
    )

    pp_formulas_dict = build_pp_formulas(config, pp_goals)
    formulas = [pp_formulas_dict[name] for name in result.formula_names]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = pp_supply(config)
    assert np.all(consumption @ result.rates <= supply + 1e-6)


def test_refine_1p4_headroom_lost_is_list_of_valid_names():
    config = WulingConfig1p4()
    base_result, base_names = search(config)
    pp_goals = PPGoals1p4()
    result = refine_1p4(
        base_result, base_names, config, pp_goals, SearchConfig(iterations=1000, seed=1)
    )
    assert isinstance(result.headroom_lost, list)
    assert all(name in ALL_NAMES for name in result.headroom_lost)


def test_refine_1p4_dollar_output_matches_rates_dot_dollar_earner_outputs():
    config = WulingConfig1p4()
    base_result, base_names = search(config)
    pp_goals = PPGoals1p4()
    result = refine_1p4(
        base_result, base_names, config, pp_goals, SearchConfig(iterations=50, seed=0)
    )

    outputs = np.array(
        [DOLLAR_EARNER_OUTPUTS.get(name, 0.0) for name in result.formula_names],
        dtype=float,
    )
    assert np.isclose(result.dollar_output, result.rates @ outputs)


def test_refine_1p4_reaches_near_dollar_target():
    config = WulingConfig1p4()
    base_result, base_names = search(config)
    pp_goals = PPGoals1p4()
    result = refine_1p4(
        base_result, base_names, config, pp_goals, SearchConfig(iterations=4000, seed=0)
    )
    assert result.dollar_output >= pp_goals.dollar_target * 0.95
