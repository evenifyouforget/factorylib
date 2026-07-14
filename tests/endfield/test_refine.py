from fractions import Fraction

import numpy as np

from factorylib.endfield.goals import WulingGoals, fitness, plan_from_search_result
from factorylib.endfield.refine import refine
from factorylib.endfield.wuling import (
    XI_PER_FORGE,
    WulingConfig,
    build_formulas,
    search,
)
from factorylib.search import SearchConfig


def test_refine_sa_improves_on_lp_optimal_fitness():
    """Regression of the empirical backend comparison in refine.py's
    module docstring: on 1.2e full, sa should find a higher-fitness plan
    than the raw LP optimum by trading some $ for simpler fractions."""
    base = search(WulingConfig())
    goals = WulingGoals()
    baseline_fitness = fitness(plan_from_search_result(base), goals)

    result = refine(
        base,
        WulingConfig(),
        goals,
        SearchConfig(iterations=3000, seed=42),
        backend="sa",
    )
    assert result.fitness > baseline_fitness


def test_refine_sa_simplifies_fractions():
    base = search(WulingConfig())
    goals = WulingGoals()
    result = refine(
        base,
        WulingConfig(),
        goals,
        SearchConfig(iterations=3000, seed=42),
        backend="sa",
    )
    for rate in result.rates:
        if abs(rate) > 1e-9:
            denom = Fraction(float(rate)).limit_denominator(1000).denominator
            assert denom <= 96  # never worse than the LP-optimal plan's worst denom


def test_refine_sa_result_is_feasible():
    config = WulingConfig()
    base = search(config)
    goals = WulingGoals()
    result = refine(
        base, config, goals, SearchConfig(iterations=1000, seed=1), backend="sa"
    )

    formulas_dict = build_formulas(config)
    formulas_dict["hx"].limit = config.max_forges - base.z
    formulas = [formulas_dict[name] for name in result.formula_names]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = config.base_supply + base.z * XI_PER_FORGE + base.metatransfer
    assert np.all(consumption @ result.rates <= supply + 1e-6)


def test_refine_scipy_backend_never_beats_sa_on_1p2e_full():
    """Documents why "sa" is the default (see refine.py's module
    docstring): scipy's dual_annealing doesn't improve on the LP optimum
    on this problem, across several seeds."""
    base = search(WulingConfig())
    goals = WulingGoals()
    baseline_fitness = fitness(plan_from_search_result(base), goals)

    for seed in range(3):
        result = refine(
            base,
            WulingConfig(),
            goals,
            SearchConfig(iterations=500, seed=seed),
            backend="scipy",
        )
        assert result.fitness <= baseline_fitness + 1e-6


def test_refine_dollar_output_matches_rates_dot_outputs():
    config = WulingConfig()
    base = search(config)
    goals = WulingGoals()
    result = refine(
        base, config, goals, SearchConfig(iterations=50, seed=0), backend="sa"
    )

    formulas_dict = build_formulas(config)
    formulas_dict["hx"].limit = config.max_forges - base.z
    outputs = np.array(
        [formulas_dict[name].output for name in result.formula_names], dtype=float
    )
    assert np.isclose(result.dollar_output, result.rates @ outputs)
