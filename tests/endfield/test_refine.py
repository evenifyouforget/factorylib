import numpy as np

from factorylib.endfield.goals import ProductionPlan, _plan_complexity
from factorylib.endfield.pp_goals import (
    ALL_NAMES,
    DOLLAR_EARNER_OUTPUTS,
    PPGoals,
    build_pp_formulas,
    pp_supply,
)
from factorylib.endfield.refine import refine
from factorylib.endfield.wuling import WulingConfig, search
from factorylib.search import SearchConfig


def test_refine_never_returns_worse_than_the_initial_baseline():
    """Structural guarantee of factorylib.search.simulated_annealing
    itself (it only ever updates its tracked best on a strict
    improvement): refine()'s returned fitness must be >= the fitness of
    its own starting point (the $-only LP-optimal plan, extended with
    zero rates for every pp-tier/bonus/delivery-quota formula it never
    ran -- so pp=0 there, and fitness is just -complexity_weight *
    complexity of that plan's own fractions)."""
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()
    baseline_plan = ProductionPlan(
        dollar_rate=0.0,
        multiples=dict(zip(base.formula_names, base.result.formula_rates)),
    )
    baseline_fitness = -pp_goals.complexity_weight * _plan_complexity(
        baseline_plan, pp_goals.max_denom
    )

    result = refine(
        base, config, pp_goals, SearchConfig(iterations=3000, seed=42), backend="sa"
    )
    assert result.fitness >= baseline_fitness


def test_refine_higher_complexity_weight_finds_lower_complexity():
    """Raising complexity_weight should trade some pp for a simpler
    plan -- a robust, weight-sensitive claim (unlike a fixed denominator
    bound, which doesn't hold up well once the search space is this much
    larger than the old $-only-formula-set fitness function's)."""
    config = WulingConfig()
    base = search(config)

    def total_complexity(pp_goals):
        result = refine(
            base, config, pp_goals, SearchConfig(iterations=3000, seed=42), backend="sa"
        )
        plan = ProductionPlan(
            dollar_rate=0.0, multiples=dict(zip(result.formula_names, result.rates))
        )
        return _plan_complexity(plan, pp_goals.max_denom)

    strict = total_complexity(PPGoals(complexity_weight=1.0))
    relaxed = total_complexity(PPGoals(complexity_weight=0.05))
    assert strict <= relaxed


def test_refine_sa_result_is_feasible():
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()
    result = refine(
        base, config, pp_goals, SearchConfig(iterations=1000, seed=1), backend="sa"
    )

    pp_formulas_dict = build_pp_formulas(config, pp_goals)
    formulas = [pp_formulas_dict[name] for name in result.formula_names]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = pp_supply(config)
    assert np.all(consumption @ result.rates <= supply + 1e-6)


def test_refine_scipy_backend_underperforms_sa_on_1p2e_full():
    """Documents why "sa" is the default (see refine.py's module
    docstring): unlike the old $-only-formula-set fitness function
    (where every dimension was already fully saturated, so scipy's
    continuous perturbation had no room to explore at all), the
    pp-scored formula set adds many dimensions with genuine slack (every
    pp-tier/delivery-quota formula starts at rate 0, real room to grow),
    so scipy's dual_annealing DOES improve on the trivial baseline now
    -- but "sa" still finds a substantially better plan across seeds."""
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()

    for seed in range(3):
        scipy_result = refine(
            base,
            config,
            pp_goals,
            SearchConfig(iterations=500, seed=seed),
            backend="scipy",
        )
        sa_result = refine(
            base,
            config,
            pp_goals,
            SearchConfig(iterations=500, seed=seed),
            backend="sa",
        )
        assert sa_result.fitness > scipy_result.fitness * 2


def test_refine_headroom_lost_is_list_of_valid_names():
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()
    result = refine(
        base, config, pp_goals, SearchConfig(iterations=1000, seed=1), backend="sa"
    )
    assert isinstance(result.headroom_lost, list)
    assert all(name in ALL_NAMES for name in result.headroom_lost)


def test_refine_dollar_output_matches_rates_dot_dollar_earner_outputs():
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()
    result = refine(
        base, config, pp_goals, SearchConfig(iterations=50, seed=0), backend="sa"
    )

    outputs = np.array(
        [DOLLAR_EARNER_OUTPUTS.get(name, 0.0) for name in result.formula_names],
        dtype=float,
    )
    assert np.isclose(result.dollar_output, result.rates @ outputs)
