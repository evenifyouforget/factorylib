"""Part 5: search for a more-fit alternative to the LP-optimal plan.

factorylib.endfield.wuling.search() maximizes raw $ only. This refines
that solution by searching for a higher-scoring nearby plan under
factorylib.endfield.pp_goals' prosperity-points system: total pp earned
minus a complexity_weight-scaled fraction-simplicity penalty (see
pp_goals' module docstring for how pp itself is computed). Since pp is
already a plain linear function of formula rates, the search's job is
narrower than it used to be: find a good simplification of an
already-near-pp-optimal allocation, not search the allocation itself
from scratch -- factorylib.search's moves (round a rate down/up to a
simpler fraction, allocate freed/unused slack, shift a resource between
two competing consumers, or re-solve the underlying $-style LP with some
rates pinned) can freely trade pp for simplicity, but round_down/shift
never discard pp for nothing (they verify network-wide feasibility
before accepting a move -- see factorylib.search's module docstring).
RefinedResult.headroom_lost flags resources that end up fully saturated
but weren't in the LP optimum -- see factorylib.search's module
docstring for what that does and doesn't prove.

The search operates over the FULL pp-scored formula set (every real
recipe formula from build_formulas(), extended with the pp-tier/bonus/
delivery-quota formulas -- see pp_goals.build_pp_formulas()), starting
from the $-only baseline's rates extended with zero for every new
pp-tier formula (the $-only LP never ran them at all).

Backend choice: factorylib.search offers both the discrete-move simulated
annealing ("sa") and a continuous scipy.optimize.dual_annealing backend
("scipy") for comparison. "sa" is the default: a continuous global
optimizer has no way to prefer small-denominator fractions except
through the complexity penalty itself, and it's still substantially
outperformed by "sa" across seeds on this scenario (see
test_refine_scipy_backend_underperforms_sa_on_1p2e_full) -- though
unlike the old $-only-formula-set fitness function (where every
dimension was already fully saturated, leaving no room for a continuous
perturbation to explore at all), the pp-scored formula set adds many
dimensions with genuine slack (every pp-tier/delivery-quota formula
starts at rate 0), so "scipy" is no longer stuck exactly at its
starting point either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorylib.endfield import pp_goals_1p4 as pg4
from factorylib.endfield import wuling_1p4 as v1p4
from factorylib.endfield.goals import ProductionPlan, _plan_complexity
from factorylib.endfield.pp_goals import (
    ALL_NAMES,
    DOLLAR_EARNER_OUTPUTS,
    PPGoals,
    build_pp_formulas,
    is_pp_bookkeeping_formula,
    pp_supply,
)
from factorylib.endfield.wuling import SearchResult, WulingConfig
from factorylib.optimize import OptimizeResult
from factorylib.search import SearchConfig, search


@dataclass
class RefinedResult:
    """Result of refine(). rates/formula_names align positionally, as in
    SearchResult -- but formula_names now includes every pp-tier/bonus/
    delivery-quota formula too (see pp_goals.build_pp_formulas), not
    just the real recipe formulas base.formula_names covers.

    Attributes:
        rates: the found plan's rate for every formula.
        pp_output: total prosperity points earned (before the
            complexity penalty) -- the pp system's own objective.
        dollar_output: real $/min recovered, computed from the
            dollar-earning formulas' true $ prices (not pp) -- what a
            player would actually see on their Wuling Stock Bill.
        fitness: pp_output minus complexity_weight * complexity(rates)
            -- the actual score the search optimizes.
        formula_names: names aligned with rates.
        headroom_lost: resource names (from RESOURCE_NAMES) that had
            spare capacity under the $-only LP-optimal plan but are
            fully saturated under rates -- see
            factorylib.search.headroom_loss(). A diagnostic, not a
            rejection.
    """

    rates: np.ndarray
    pp_output: float
    dollar_output: float
    fitness: float
    formula_names: list[str]
    headroom_lost: list[str]


def refine(
    base: SearchResult,
    wuling_config: WulingConfig,
    pp_goals: PPGoals,
    search_config: SearchConfig | None = None,
    *,
    backend: str = "sa",
) -> RefinedResult:
    """Search for a high-pp, low-complexity nearby plan, starting from
    base (an LP-optimal SearchResult from
    factorylib.endfield.wuling.search()) extended with zero rates for
    every pp-tier/bonus/delivery-quota formula the $-only LP never ran.
    """
    pp_formulas_dict = build_pp_formulas(wuling_config, pp_goals)
    formula_names = list(pp_formulas_dict.keys())
    formulas = [pp_formulas_dict[name] for name in formula_names]
    pp_outputs = np.array([f.output for f in formulas], dtype=float)
    dollar_outputs = np.array(
        [DOLLAR_EARNER_OUTPUTS.get(name, 0.0) for name in formula_names], dtype=float
    )
    supply = pp_supply(wuling_config)
    consumption_by_name = {name: f.consumption for name, f in pp_formulas_dict.items()}

    base_rates_by_name = dict(zip(base.formula_names, base.result.formula_rates))
    initial_rates = np.array(
        [base_rates_by_name.get(name, 0.0) for name in formula_names], dtype=float
    )

    def fitness_fn(rates: np.ndarray) -> float:
        pp = float(rates @ pp_outputs)
        plan = ProductionPlan(
            dollar_rate=0.0,
            multiples=dict(zip(formula_names, rates)),
            consumption=consumption_by_name,
        )
        return pp - pp_goals.complexity_weight * _plan_complexity(
            plan, pp_goals.max_denom
        )

    outcome = search(
        supply,
        formulas,
        initial_rates,
        fitness_fn,
        search_config,
        backend=backend,
    )
    return RefinedResult(
        rates=outcome.rates,
        pp_output=float(outcome.rates @ pp_outputs),
        dollar_output=float(outcome.rates @ dollar_outputs),
        fitness=outcome.fitness,
        formula_names=formula_names,
        headroom_lost=[ALL_NAMES[k] for k in outcome.headroom_lost],
    )


def _is_1p4_bookkeeping_formula(name: str) -> bool:
    return v1p4.is_forge_or_metatransfer_formula(name) or is_pp_bookkeeping_formula(
        name
    )


def refine_1p4(
    base_result: OptimizeResult,
    base_formula_names: list[str],
    wuling_config: v1p4.WulingConfig1p4,
    pp_goals: pg4.PPGoals1p4,
    search_config: SearchConfig | None = None,
    *,
    backend: str = "sa",
) -> RefinedResult:
    """1.4 equivalent of refine(): same fitness function (pp_output minus
    a complexity-weighted fraction-simplicity penalty), wired to
    wuling_1p4/pp_goals_1p4 instead of 1.2e's wuling/pp_goals. A separate
    function rather than generalizing refine() itself, to avoid any risk
    to refine()'s own tests -- see tmp_notes/wip_todo.md.

    base_result/base_formula_names come from wuling_1p4.search(), which
    returns a plain (OptimizeResult, list[str]) tuple rather than 1.2e's
    SearchResult wrapper (1.4 has no z/metatransfer scalar bookkeeping to
    surface specially -- every discrete choice is an ordinary named
    formula rate, see wuling_1p4.search's own docstring), hence the two
    separate parameters instead of one SearchResult.
    """
    pp_formulas_dict = pg4.build_pp_formulas(wuling_config, pp_goals)
    formula_names = list(pp_formulas_dict.keys())
    formulas = [pp_formulas_dict[name] for name in formula_names]
    pp_outputs = np.array([f.output for f in formulas], dtype=float)
    dollar_outputs = np.array(
        [pg4.DOLLAR_EARNER_OUTPUTS.get(name, 0.0) for name in formula_names],
        dtype=float,
    )
    supply = pg4.pp_supply(wuling_config)
    consumption_by_name = {name: f.consumption for name, f in pp_formulas_dict.items()}

    base_rates_by_name = dict(zip(base_formula_names, base_result.formula_rates))
    initial_rates = np.array(
        [base_rates_by_name.get(name, 0.0) for name in formula_names], dtype=float
    )

    def fitness_fn(rates: np.ndarray) -> float:
        pp = float(rates @ pp_outputs)
        plan = ProductionPlan(
            dollar_rate=0.0,
            multiples=dict(zip(formula_names, rates)),
            consumption=consumption_by_name,
        )
        complexity = _plan_complexity(
            plan,
            pp_goals.max_denom,
            resource_names=v1p4.RESOURCE_NAMES,
            resource_belt_speed=v1p4.RESOURCE_BELT_SPEED,
            is_bookkeeping_formula=_is_1p4_bookkeeping_formula,
        )
        return pp - pp_goals.complexity_weight * complexity

    outcome = search(
        supply,
        formulas,
        initial_rates,
        fitness_fn,
        search_config,
        backend=backend,
    )
    return RefinedResult(
        rates=outcome.rates,
        pp_output=float(outcome.rates @ pp_outputs),
        dollar_output=float(outcome.rates @ dollar_outputs),
        fitness=outcome.fitness,
        formula_names=formula_names,
        headroom_lost=[pg4.ALL_NAMES[k] for k in outcome.headroom_lost],
    )
