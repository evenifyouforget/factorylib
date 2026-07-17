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
from factorylib.endfield.wuling import POWER_YIELD, WulingConfig, search
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


def test_refine_dollar_output_exceeds_target_on_1p2e_full():
    """Regression: with satisfaction_tiers' old pp_decay=0.15 default,
    dollar's post-ramp tiers decayed to ~3.4 pp/$ -- just under the ~6.7
    pp/$ needed to ever outbid power's tail tier for another unit of
    shared battery capacity, so the search deterministically settled at
    EXACTLY 100% of the dollar target across every seed and iteration
    count tried (0..60000), even though the $-only LP baseline shows
    more is physically available. Confirmed via direct CLI testing this
    looked wrong ("massively overcommits batteries to power... doesn't
    produce any excess sellable goods"). pp_decay=0.20 (see
    satisfaction_tiers' own docstring for the concrete numbers) fixes
    this: dollar output should exceed its target, not just equal it,
    while power/delivery/some Gear Components remain satisfied too."""
    config = WulingConfig()
    base = search(config)
    pp_goals = PPGoals()
    result = refine(
        base, config, pp_goals, SearchConfig(iterations=6000, seed=0), backend="sa"
    )

    assert result.dollar_output > pp_goals.dollar_target

    rates = dict(zip(result.formula_names, result.rates))
    power = sum(rate * POWER_YIELD.get(name, 0.0) for name, rate in rates.items())
    quota = sum(
        rate
        for name, rate in rates.items()
        if name.startswith("delivery_quota_from_") and rate > 1e-9
    )
    assert power > pp_goals.power_target
    assert quota > pp_goals.delivery_jobs_per_day
    assert rates.get("cuprium_component", 0.0) > 0.0
    assert rates.get("xiranite_component", 0.0) > 0.0


def test_refine_lower_power_hard_cap_frees_more_dollar_headroom():
    """Regression for tmp_notes/adjust_power_curve.md: there's no real
    justification for tolerating 40% power overshoot (a battery's charge
    cycle only wastes <1% of its energy near realistic demand levels,
    and DIGE's ~5-10% overshoot convention already covers essentially
    all practical cases) -- power_hard_cap_ratio=1.40 just let the
    search tie up battery capacity in power well past the point of
    diminishing returns, capacity that has better uses (more sellable
    $ output) once freed. Confirms a materially tighter hard cap (10%)
    yields strictly more dollar output than a looser one (40%), with
    power still comfortably above its own target either way -- a
    concrete "is this actually better" check, not just "does it still
    run"."""
    config = WulingConfig()
    base = search(config)
    search_config = SearchConfig(iterations=6000, seed=0)

    loose = refine(
        base, config, PPGoals(power_hard_cap_ratio=1.40), search_config, backend="sa"
    )
    tight = refine(
        base, config, PPGoals(power_hard_cap_ratio=1.10), search_config, backend="sa"
    )
    assert tight.dollar_output > loose.dollar_output
