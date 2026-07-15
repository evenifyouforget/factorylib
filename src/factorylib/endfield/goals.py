"""Expanded Wuling goals: a nonlinear fitness function over a whole
production plan (Part 4).

Beyond raw $ (Wuling Stock Bill) income, a good production plan should also:
  - stay comfortably (not just barely) above the savings-limited sell rate,
  - produce enough power (batteries),
  - produce a couple of cheap "delivery job" filler materials, and
  - produce a small surplus of gear-crafting materials, in priority order
    (Parts > Components > Bottles > Xiranite/Heavy Xiranite > plants > other),
while preferring simple (small-denominator, small-prime) production
fractions over complex ones, even at a small cost to raw output.

Design alternatives considered for combining these into one score:
  - Plain weighted linear sum of raw quantities: simplest, but lets a
    large surplus in one goal "buy out" a shortfall in another (e.g. tons
    of excess $ compensating for zero power), which doesn't match "we
    always want at least this much power" reading as closer to a soft
    floor than a freely tradeable quantity.
  - Strict lexicographic ordering (fully satisfy goal N before goal N+1
    is considered at all): matches the stated priority order literally,
    but explicitly contradicted by the spec, which wants trade-offs (e.g.
    "slightly less than max sellable goods, in order to have simple
    denominator fractions").
  - Chosen: per-goal *shaped* (piecewise/logarithmic) terms, summed with
    per-category weights. Each term steeply penalizes falling short of
    its target, rewards closing the gap up to a comfortable margin, and
    then flattens (diminishing log return) past that margin -- so more
    is never penalized, but stops being worth much once "enough," freeing
    up capacity for lower-priority goals. Priority ordering falls out of
    the per-category weights rather than needing a different functional
    shape per goal, keeping the function itself simple.

Power, delivery-job filler, and four of the six gear Components are
modeled (as zero-$ formulas -- see factorylib.endfield.wuling's module
docstring for exactly what's covered and what's still missing, e.g.
Cryston/Amethyst Components and battery-diverted power). Since Formula
carries only a $ output, power is tracked separately via
factorylib.endfield.wuling.POWER_YIELD; plan_from_search_result() reads
it off directly. good_rates is every formula's rate by name, converted
from raw recipe multiples to real items/min via
factorylib.endfield.wuling.GOOD_YIELD where applicable (e.g.
sandleaf_powder's rate is multiples of "-> 90 Sandleaf Powder", so 1.0
multiples/min means 90 items/min) -- WulingGoals.delivery_goods and
gear_min_target are both stated in items/min, matching the spec's own
units ("0.5/min of Cuprium Component"), so this conversion has to
happen before the comparison, not after.

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
not just its rate -- see ProductionPlan.consumption.

Note this only proves LP-level feasibility (every accepted plan
satisfies consumption @ rates <= supply for every tracked resource,
including cyclic ones like sewage/effluent -- see factorylib.search's
moves). It says nothing about physical topology or priority-splitter
dynamics (transient clogging, depot turn-taking, backpressure-driven
"auto-limiting" where a ratio-limited co-reactant caps the achieved flow
below nominal belt capacity) -- those aren't modeled here at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factorylib.endfield.wuling import (
    GOOD_YIELD,
    POWER_YIELD,
    RESOURCE_BELT_SPEED,
    RESOURCE_NAMES,
    SearchResult,
    WulingConfig,
    build_formulas,
)
from factorylib.simplicity import fraction_complexity


@dataclass
class WulingGoals:
    """Configurable targets for the expanded fitness function.

    Args:
        stock_bill_cap: max $/min actually sellable, limited by savings
            generation (Sky King Flats Construction Site 700/min +
            Cardiac Remediation Station 390/min, by default). Likely to
            increase in future updates -- configurable for that reason.
        stock_bill_target_ratio: desired excess over the cap (>=1.0 is a
            hard floor -- below it, savings drains to 0 over many sell
            cycles; the spec's own example uses 1.1x as still-too-slow-ish
            and prefers more, so the default target sits a bit above that).
        stock_bill_importance: overall weight of the stock-bill term.
        power_target: W worth of batteries to produce (average player
            demand, ~7000 today per the spec).
        power_importance: overall weight of the power term.
        delivery_goods: name -> minimum acceptable production rate (per
            the same time unit as other rates) for delivery-job filler
            materials. Defaults to sandleaf_powder (the only delivery
            filler currently modeled) at 15/min -- well above the ~9.7/min
            floor implied by "more than 14k in 24h", with a safety margin.
        delivery_importance: overall weight of each delivery-good term.
        gear_priority: formula/good names in descending demand priority.
            Defaults to the four Gear Components currently modeled (see
            factorylib.endfield.wuling's module docstring for which ones
            and why); the spec's fuller "Parts > Components > Bottles >
            Xiranite/Heavy Xiranite > plants" ordering doesn't rank the
            Components against each other, so this default order is a
            reasonable placeholder, not a specified ranking.
        gear_min_target: minimum acceptable rate for each gear material
            (spec: "even 0.5/min of Cuprium Component" is already ample).
        gear_importance: overall weight of the highest-priority gear term.
        gear_priority_decay: multiplier applied per rank down
            gear_priority (rank 0 keeps gear_importance; rank k gets
            gear_importance * gear_priority_decay**k), implementing the
            demand ordering as a soft priority rather than a hard one.
        complexity_weight: weight of the simplicity/denominator penalty
            (factorylib.simplicity.fraction_complexity) applied to every
            nonzero formula multiple in the plan.
        max_denom: denominator ceiling used when pricing a rate's
            complexity (see factorylib.simplicity.fraction_complexity).
    """

    stock_bill_cap: float = 1090.0
    stock_bill_target_ratio: float = 1.15
    stock_bill_importance: float = 20.0
    power_target: float = 7000.0
    power_importance: float = 10.0
    delivery_goods: dict[str, float] = field(
        default_factory=lambda: {"sandleaf_powder": 15.0}
    )
    delivery_importance: float = 3.0
    gear_priority: list[str] = field(
        default_factory=lambda: [
            "hetonite_component",
            "xiranite_component",
            "cuprium_component",
            "ferrium_component",
        ]
    )
    gear_min_target: float = 0.5
    gear_importance: float = 2.0
    gear_priority_decay: float = 0.5
    complexity_weight: float = 1.0
    max_denom: int = 1000


@dataclass
class ProductionPlan:
    """An overall production plan to be scored by fitness().

    Attributes:
        dollar_rate: $/min (Wuling Stock Bill sell rate).
        power_rate: W worth of power produced.
        good_rates: named production rates for delivery/gear-relevant
            materials, in items/min (see item_rates()), keyed to match
            WulingGoals.delivery_goods and gear_priority. Goods not
            actually modeled in the current recipe set simply aren't
            present as keys, and fitness() treats a missing key as rate
            0.0.
        multiples: every formula's rate (by name), used for the
            simplicity/denominator penalty.
        consumption: each formula's resource-consumption vector (by
            name), used to price complexity per physical belt/pipe flow
            instead of per raw rate (see module docstring). A formula
            missing from this dict falls back to pricing its raw rate
            directly, so callers that only care about multiples-based
            complexity (e.g. synthetic test plans) don't need to supply
            it.
    """

    dollar_rate: float
    power_rate: float = 0.0
    good_rates: dict[str, float] = field(default_factory=dict)
    multiples: dict[str, float] = field(default_factory=dict)
    consumption: dict[str, np.ndarray] = field(default_factory=dict)


def item_rates(multiples: dict[str, float]) -> dict[str, float]:
    """Convert each formula's raw rate (recipe multiples/min) to a real
    item rate (items/min) via GOOD_YIELD, for formulas that produce a
    named, stashable good at $0 output. Formulas with no GOOD_YIELD entry
    keep their raw rate unchanged (harmless: WulingGoals only looks up
    names it actually cares about)."""
    return {name: rate * GOOD_YIELD.get(name, 1.0) for name, rate in multiples.items()}


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
    if not config.fix_hx_limit and "hx_make" in formulas_dict:
        formulas_dict["hx_make"].limit = config.max_forges - result.z
    consumption = {
        name: formulas_dict[name].consumption
        for name in result.formula_names
        if name in formulas_dict
    }
    return ProductionPlan(
        dollar_rate=result.result.dollar_output,
        power_rate=power_rate,
        good_rates=item_rates(multiples),
        multiples=multiples,
        consumption=consumption,
    )


def _threshold_term(
    rate: float, target: float, *, shortfall_penalty: float, excess_gain: float
) -> float:
    """Score rate against target: quadratically penalize falling short
    (steeper the closer to a total shortfall), then reward any excess only
    mildly, with diminishing (logarithmic) returns. target <= 0 means "no
    requirement" (e.g. a good that isn't modeled/relevant yet) -> 0."""
    if target <= 0:
        return 0.0
    ratio = rate / target
    if ratio < 1.0:
        return -shortfall_penalty * (1.0 - ratio) ** 2
    return excess_gain * math.log1p(ratio - 1.0)


def _stock_bill_term(rate: float, goals: WulingGoals) -> float:
    """Score the $/min rate against the savings-limited cap.

    Three pieces: quadratic penalty below the cap (a hard floor -- below
    it, savings drains toward 0 over many sell cycles); strong, roughly
    linear reward climbing from the cap up to the target excess ratio
    (the "comfortable margin" the spec wants); then a much flatter
    logarithmic reward beyond that, since further excess barely helps and
    that capacity is better spent on other goals.
    """
    cap = goals.stock_bill_cap
    if cap <= 0:
        return 0.0
    ratio = rate / cap
    shortfall_penalty = goals.stock_bill_importance * 10.0
    growth_gain = goals.stock_bill_importance
    excess_gain = goals.stock_bill_importance * 0.1
    if ratio < 1.0:
        return -shortfall_penalty * (1.0 - ratio) ** 2
    if ratio < goals.stock_bill_target_ratio:
        return growth_gain * (ratio - 1.0)
    plateau = growth_gain * (goals.stock_bill_target_ratio - 1.0)
    return plateau + excess_gain * math.log1p(ratio - goals.stock_bill_target_ratio)


def fitness(plan: ProductionPlan, goals: WulingGoals) -> float:
    """Score a production plan. Higher is better. See module docstring for
    the design rationale."""
    score = _stock_bill_term(plan.dollar_rate, goals)

    score += _threshold_term(
        plan.power_rate,
        goals.power_target,
        shortfall_penalty=goals.power_importance * 10.0,
        excess_gain=goals.power_importance * 0.1,
    )

    for name, target in goals.delivery_goods.items():
        score += _threshold_term(
            plan.good_rates.get(name, 0.0),
            target,
            shortfall_penalty=goals.delivery_importance * 10.0,
            excess_gain=goals.delivery_importance * 0.1,
        )

    for rank, name in enumerate(goals.gear_priority):
        weight = goals.gear_importance * (goals.gear_priority_decay**rank)
        score += _threshold_term(
            plan.good_rates.get(name, 0.0),
            goals.gear_min_target,
            shortfall_penalty=weight * 10.0,
            excess_gain=weight * 0.1,
        )

    score -= goals.complexity_weight * _plan_complexity(plan, goals.max_denom)

    return score


def _plan_complexity(plan: ProductionPlan, max_denom: int) -> float:
    """Total complexity penalty for a plan: for each formula with a known
    consumption vector, price every nonzero resource flow it induces as a
    belt/pipe-fraction (see module docstring); formulas with no known
    consumption vector fall back to pricing their raw rate directly.
    """
    total = 0.0
    for name, rate in plan.multiples.items():
        if abs(rate) < 1e-9:
            continue
        consumption = plan.consumption.get(name)
        if consumption is None:
            total += fraction_complexity(rate, max_denom)
            continue
        for resource_name, coeff in zip(RESOURCE_NAMES, consumption):
            if abs(coeff) < 1e-12:
                continue
            belt_speed = RESOURCE_BELT_SPEED.get(resource_name)
            if not belt_speed:
                total += fraction_complexity(rate * coeff, max_denom)
                continue
            belts = rate * coeff / belt_speed
            total += fraction_complexity(belts, max_denom)
    return total


# Craft Gear: "8000 Wuling Stock Bill + 50 Xiranite Component -> 1
# Xiranite Component Gear" etc. Deliberately NOT modeled as a Formula
# (see factorylib.endfield.wuling's module docstring): it spends
# *accumulated* Stock Bill and Component items -- a one-time stock, not
# a steady per-minute flow -- which doesn't fit this LP's steady-state
# framework at all. name -> (Stock Bill cost, Component cost, Gear name).
GEAR_RECIPES: dict[str, tuple[float, float, str]] = {
    "xiranite_component": (8000.0, 50.0, "Xiranite Component Gear"),
    "cuprium_component": (16000.0, 50.0, "Cuprium Component Gear"),
    "hetonite_component": (25000.0, 50.0, "Hetonite Component Gear"),
}

_MINUTES_PER_DAY = 24 * 60


def days_to_afford_gear(
    component_name: str, sold_dollar_rate: float, component_item_rate: float
) -> float | None:
    """Days until enough Wuling Stock Bill (accumulated at
    sold_dollar_rate -- the outpost-savings-capped *sold* $/min, not raw
    produced $/min; see factorylib.priority_sell) and the named Component
    (accumulated at component_item_rate items/min) have both piled up
    to Craft this Gear, assuming neither is spent on anything else in the
    meantime. Returns None if either rate is non-positive (would never
    accumulate at all, so there's no finite answer).
    """
    if component_name not in GEAR_RECIPES:
        raise ValueError(f"Unknown gear recipe: {component_name!r}")
    if sold_dollar_rate <= 0 or component_item_rate <= 0:
        return None
    stock_bill_cost, component_cost, _ = GEAR_RECIPES[component_name]
    days_for_stock_bill = stock_bill_cost / (sold_dollar_rate * _MINUTES_PER_DAY)
    days_for_component = component_cost / (component_item_rate * _MINUTES_PER_DAY)
    return max(days_for_stock_bill, days_for_component)
