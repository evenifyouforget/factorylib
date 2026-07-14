"""Part 5: search for a more-fit alternative to the LP-optimal plan.

factorylib.endfield.wuling.search() maximizes raw $ only. This refines
that solution using factorylib.search's local search, scoring candidates
with factorylib.endfield.goals.fitness instead -- which can prefer a plan
with slightly less $ but simpler fractions, or a bit of production
diverted to secondary goals (see Part 4). It starts from the LP optimum,
so the three moves (round a rate down to a simpler fraction; round a
rate up to a simpler fraction if the slack allows; allocate freed/unused
slack to another formula) can only trade $ for simplicity/secondary
goals, never discard money for nothing. RefinedResult.headroom_lost
flags resources that end up fully saturated but weren't in the LP
optimum -- see factorylib.search's module docstring for what that does
and doesn't prove.

Backend choice: factorylib.search offers both the discrete-move simulated
annealing ("sa") and a continuous scipy.optimize.dual_annealing backend
("scipy") for comparison. Tried on the 1.2e-full scenario (default
WulingConfig/WulingGoals, several seeds): "sa" improved fitness from the
LP-optimal plan's -218.7 to -164.5, trading ~$162/min for an all-integer
solution that also picks up the two secondary goals cheap enough to be
worth it here (Sandleaf Powder and Thermal Bank, both of which compete
for little or nothing against the $-formulas). "scipy" never improved on
the LP-optimal plan at all -- that plan turns out to be a fully
resource-saturated LP vertex (zero slack in every resource dimension), so
*any* continuous perturbation away from it immediately violates some
constraint; the penalty term in scipy_dual_annealing's objective drives
the search right back to the starting point every time, so it never
explores at all on this problem. "sa"'s discrete moves don't have this
issue since round_down always frees slack before allocate_slack tries to
spend it. "sa" is therefore the default; "scipy" is kept available for
comparison, not because it currently wins here.

Note "sa" does not pick up the Gear Component goals on this scenario: at
default weights, even sacrificing an entire SC Wuling Battery run
(-$324/min) to fully satisfy the Ferrium Component floor still scores
worse than not bothering (Originium Ore and Ferrium Ore are simply too
valuable to the existing $-formulas already using them). This isn't a
search bug -- fitness() genuinely disprefers that trade at these
weights; raising WulingGoals.gear_importance (or lowering
stock_bill_importance) will change that trade-off if a stronger gear
guarantee is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorylib.endfield.goals import ProductionPlan, WulingGoals, fitness, item_rates
from factorylib.endfield.wuling import (
    POWER_YIELD,
    RESOURCE_NAMES,
    XI_PER_FORGE,
    SearchResult,
    WulingConfig,
    build_formulas,
)
from factorylib.search import SearchConfig, search


@dataclass
class RefinedResult:
    """Result of refine(). rates/formula_names align positionally, as in
    SearchResult.

    headroom_lost: resource names (from RESOURCE_NAMES) that had spare
    capacity under the LP-optimal plan but are fully saturated under
    rates -- see factorylib.search.headroom_loss(). A diagnostic, not a
    rejection: not necessarily a problem, but worth a human glance (see
    factorylib.endfield.goals's module docstring for what this does and
    doesn't prove).
    """

    rates: np.ndarray
    dollar_output: float
    fitness: float
    formula_names: list[str]
    headroom_lost: list[str]


def _plan_from_rates(
    rates: np.ndarray,
    formula_names: list[str],
    original_outputs: np.ndarray,
    consumption: dict[str, np.ndarray],
) -> ProductionPlan:
    multiples = dict(zip(formula_names, rates))
    power_rate = sum(
        rate * POWER_YIELD.get(name, 0.0) for name, rate in multiples.items()
    )
    return ProductionPlan(
        dollar_rate=float(np.asarray(rates) @ original_outputs),
        power_rate=power_rate,
        good_rates=item_rates(multiples),
        multiples=multiples,
        consumption=consumption,
    )


def refine(
    base: SearchResult,
    wuling_config: WulingConfig,
    goals: WulingGoals,
    search_config: SearchConfig | None = None,
    *,
    backend: str = "sa",
) -> RefinedResult:
    """Search for a more-fit nearby plan than base (an LP-optimal
    SearchResult from factorylib.endfield.wuling.search()), at the same
    forge allocation (z) and metatransfer base already chosen there."""
    formulas_dict = build_formulas(wuling_config)
    if not wuling_config.fix_hx_limit:
        formulas_dict["hx"].limit = wuling_config.max_forges - base.z
    formulas = [formulas_dict[name] for name in base.formula_names]
    original_outputs = np.array([f.output for f in formulas], dtype=float)
    supply = wuling_config.base_supply + base.z * XI_PER_FORGE + base.metatransfer
    consumption_by_name = dict(
        zip(base.formula_names, (f.consumption for f in formulas))
    )

    def fitness_fn(rates: np.ndarray) -> float:
        plan = _plan_from_rates(
            rates, base.formula_names, original_outputs, consumption_by_name
        )
        return fitness(plan, goals)

    outcome = search(
        supply,
        formulas,
        base.result.formula_rates,
        fitness_fn,
        search_config,
        backend=backend,
    )
    return RefinedResult(
        rates=outcome.rates,
        dollar_output=float(outcome.rates @ original_outputs),
        fitness=outcome.fitness,
        formula_names=base.formula_names,
        headroom_lost=[RESOURCE_NAMES[k] for k in outcome.headroom_lost],
    )
