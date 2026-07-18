"""Production-plan snapshotting and fraction-complexity pricing (Part 4).

The nonlinear, hand-shaped fitness function that used to live here
(WulingGoals/fitness()/_threshold_term/_stock_bill_term -- per-goal
shaped curves summed with per-category weights) has been replaced by
factorylib.endfield.pp_goals' prosperity-points system: every goal
becomes ordinary Formula entries with strictly-decreasing pp-per-unit
slope, so a plain LP naturally reproduces the same "diminishing returns
above a target" shape with zero nonlinear solver machinery, and
factorylib.endfield.refine.refine()'s own fitness function is just
`pp_output - complexity_weight * complexity(rates)` -- see pp_goals'
module docstring for the full design rationale.

What's left here is genuinely goal-agnostic and still needed either
way: a snapshot of a production plan (ProductionPlan), converting a
formula's raw rate to a real item rate (item_rates()), finding the NET
rate of a good that has other consumers besides its own producer
(delivery_good_rate()), building a ProductionPlan from a
factorylib.endfield.wuling.search() result (plan_from_search_result()),
and pricing a plan's fraction complexity per physical resource *flow*
rather than per raw formula rate (_plan_complexity()).

Complexity is priced per physical resource *flow*, not per raw formula
rate (see factorylib_tmp_physical_factory_construction.md): belts run at
30 items/min, pipes at 120 items/min, and a formula's own "multiples"
fraction can look arbitrarily complex while every resource flow it
induces still lands on a whole belt (e.g. a recipe needing 120 items/run
of some input, run at rate 1/4, draws exactly 30 items/min of it -- one
full belt, zero splitting -- even though 1/4 alone might look like it
needs a 4-way split). Pricing the belt-fraction of each flow instead of
the raw rate gets this right in both directions: a "nice-looking" rate
can still be penalized if it happens to induce an awkward belt-split on
some resource. This requires knowing each formula's consumption vector,
not just its rate -- see ProductionPlan.consumption. Forge/metatransfer
bookkeeping formulas (wuling.is_forge_or_metatransfer_formula) and pp
tier/bonus/delivery-quota bookkeeping formulas
(pp_goals.is_pp_bookkeeping_formula) are excluded from this pricing --
neither represents a physical belt a player would ever build.

Note this only proves LP-level feasibility (every accepted plan
satisfies consumption @ rates <= supply for every tracked resource,
including cyclic ones like sewage/effluent -- see factorylib.search's
moves). It says nothing about physical topology or priority-splitter
dynamics (transient clogging, depot turn-taking, backpressure-driven
"auto-limiting" where a ratio-limited co-reactant caps the achieved flow
below nominal belt capacity) -- those aren't modeled here at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from factorylib.endfield.pp_goals import is_pp_bookkeeping_formula
from factorylib.endfield.wuling import (
    GOOD_YIELD,
    POWER_YIELD,
    RESOURCE_BELT_SPEED,
    RESOURCE_NAMES,
    SearchResult,
    WulingConfig,
    build_formulas,
    is_forge_or_metatransfer_formula,
)
from factorylib.simplicity import fraction_complexity


@dataclass
class ProductionPlan:
    """A snapshot of an overall production plan.

    Attributes:
        dollar_rate: $/min (Wuling Stock Bill sell rate).
        power_rate: W worth of power produced.
        good_rates: named production rates for delivery/gear-relevant
            materials, in items/min (see item_rates()).
        multiples: every formula's rate (by name), used for the
            simplicity/denominator penalty.
        consumption: each formula's resource-consumption vector (by
            name), used to price complexity per physical belt/pipe flow
            instead of per raw rate (see module docstring). A formula
            missing from this dict falls back to pricing its raw rate
            directly, so callers that only care about multiples-based
            complexity (e.g. synthetic test plans) don't need to supply
            it.
        resource_slack: net (production minus ALL consumption) rate for
            each tracked resource (by RESOURCE_NAMES name), used by
            delivery_good_rate() to find how much of a delivery good is
            actually left over to accumulate once its OTHER consumers
            (e.g. ori_to_dop eating into sandleaf_powder's own
            production) take their share -- good_rates alone only has
            the formula's own GROSS rate, which overstates that.
    """

    dollar_rate: float
    power_rate: float = 0.0
    good_rates: dict[str, float] = field(default_factory=dict)
    multiples: dict[str, float] = field(default_factory=dict)
    consumption: dict[str, np.ndarray] = field(default_factory=dict)
    resource_slack: dict[str, float] = field(default_factory=dict)


def item_rates(multiples: dict[str, float]) -> dict[str, float]:
    """Convert each formula's raw rate (recipe multiples/min) to a real
    item rate (items/min) via GOOD_YIELD, for formulas that produce a
    named, stashable good at $0 output. Formulas with no GOOD_YIELD entry
    keep their raw rate unchanged (harmless: callers only look up names
    they actually care about)."""
    return {name: rate * GOOD_YIELD.get(name, 1.0) for name, rate in multiples.items()}


def delivery_good_rate(
    name: str,
    consumption: dict[str, np.ndarray],
    resource_slack: dict[str, float],
    fallback_rate: float,
) -> float:
    """The rate actually available for delivery/stockpiling of `name`.

    If its formula produces exactly one resource this model tracks
    (negative consumption coefficient) and that resource's NET slack is
    known, use that -- production minus ALL consumption, e.g. ori_to_dop
    eating into sandleaf_powder's own production -- not the formula's
    own GROSS rate (fallback_rate), which overstates what's actually
    left over. Falls back to fallback_rate (typically the gross rate)
    if the formula produces no resource this model tracks at all (e.g.
    the Gear Components, which are pure terminal goods with no resource
    dimension of their own -- gross and net coincide for those anyway,
    since nothing else can consume them)."""
    vec = consumption.get(name)
    if vec is not None and resource_slack:
        produced = [i for i, c in enumerate(vec) if c < -1e-9]
        if len(produced) == 1:
            resource_name = RESOURCE_NAMES[produced[0]]
            if resource_name in resource_slack:
                return resource_slack[resource_name]
    return fallback_rate


def plan_from_search_result(
    result: SearchResult, config: WulingConfig
) -> ProductionPlan:
    """Build a ProductionPlan from a factorylib.endfield.wuling.search()
    result and the WulingConfig it was produced from (needed to rebuild
    each formula's consumption vector for belt-aware complexity pricing).
    power_rate is computed from POWER_YIELD; good_rates is every
    formula's rate by name, in items/min (see ProductionPlan.good_rates
    and item_rates())."""
    multiples = dict(zip(result.formula_names, result.result.formula_rates))
    power_rate = sum(
        rate * POWER_YIELD.get(name, 0.0) for name, rate in multiples.items()
    )
    formulas_dict = build_formulas(config)
    consumption = {
        name: formulas_dict[name].consumption
        for name in result.formula_names
        if name in formulas_dict
    }
    resource_slack = dict(zip(RESOURCE_NAMES, result.result.resource_slack))
    return ProductionPlan(
        dollar_rate=result.result.dollar_output,
        power_rate=power_rate,
        good_rates=item_rates(multiples),
        multiples=multiples,
        consumption=consumption,
        resource_slack=resource_slack,
    )


def _plan_complexity(
    plan: ProductionPlan,
    max_denom: int,
    *,
    resource_names: Sequence[str] = RESOURCE_NAMES,
    resource_belt_speed: dict[str, float] = RESOURCE_BELT_SPEED,
    is_bookkeeping_formula: Callable[[str], bool] | None = None,
) -> float:
    """Total complexity penalty for a plan: for each formula with a known
    consumption vector, price every nonzero resource flow it induces as a
    belt/pipe-fraction (see module docstring); formulas with no known
    consumption vector fall back to pricing their raw rate directly.

    resource_names/resource_belt_speed/is_bookkeeping_formula default to
    1.2e's own (wuling.py's RESOURCE_NAMES/RESOURCE_BELT_SPEED, and
    wuling.is_forge_or_metatransfer_formula combined with
    pp_goals.is_pp_bookkeeping_formula) -- pass 1.4-scenario equivalents
    (e.g. wuling_1p4.RESOURCE_NAMES/RESOURCE_BELT_SPEED and a predicate
    covering wuling_1p4's own forge-allocation formula names) for a 1.4
    plan instead. A consumption vector longer than resource_names (e.g.
    one extended to pp_goals' FLOW_NAMES) is handled correctly: zip()
    stops at the shorter resource_names, which is exactly right since
    flow dimensions aren't real physical belts to price at all.
    """
    if is_bookkeeping_formula is None:

        def is_bookkeeping_formula(name: str) -> bool:
            return is_forge_or_metatransfer_formula(name) or is_pp_bookkeeping_formula(
                name
            )

    total = 0.0
    for name, rate in plan.multiples.items():
        if abs(rate) < 1e-9 or is_bookkeeping_formula(name):
            continue
        consumption = plan.consumption.get(name)
        if consumption is None:
            total += fraction_complexity(rate, max_denom)
            continue
        for resource_name, coeff in zip(resource_names, consumption):
            if abs(coeff) < 1e-12:
                continue
            belt_speed = resource_belt_speed.get(resource_name)
            if not belt_speed:
                total += fraction_complexity(rate * coeff, max_denom)
                continue
            belts = rate * coeff / belt_speed
            total += fraction_complexity(belts, max_denom)
    return total
