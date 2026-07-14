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
it off directly. good_rates is just every formula's rate by name (the
same dict as multiples) -- WulingGoals.delivery_goods/gear_priority key
into whichever of those names are actually modeled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from factorylib.endfield.wuling import POWER_YIELD, SearchResult
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
            materials, keyed to match WulingGoals.delivery_goods and
            gear_priority. In practice this is just every formula's rate
            by name (same as multiples) -- goods not actually modeled in
            the current recipe set simply aren't present as keys, and
            fitness() treats a missing key as rate 0.0.
        multiples: every formula's rate (by name), used for the
            simplicity/denominator penalty.
    """

    dollar_rate: float
    power_rate: float = 0.0
    good_rates: dict[str, float] = field(default_factory=dict)
    multiples: dict[str, float] = field(default_factory=dict)


def plan_from_search_result(result: SearchResult) -> ProductionPlan:
    """Build a ProductionPlan from a factorylib.endfield.wuling.search()
    result. power_rate is computed from POWER_YIELD; good_rates is every
    formula's rate by name (see ProductionPlan.good_rates)."""
    multiples = dict(zip(result.formula_names, result.result.formula_rates))
    power_rate = sum(
        rate * POWER_YIELD.get(name, 0.0) for name, rate in multiples.items()
    )
    return ProductionPlan(
        dollar_rate=result.result.dollar_output,
        power_rate=power_rate,
        good_rates=multiples,
        multiples=multiples,
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

    complexity_penalty = sum(
        fraction_complexity(rate, goals.max_denom) for rate in plan.multiples.values()
    )
    score -= goals.complexity_weight * complexity_penalty

    return score
