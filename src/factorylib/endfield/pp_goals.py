"""Prosperity Points (pp): a pure-LP replacement for the Part 4 nonlinear
fitness function (see factorylib_tmp_linearization.md).

Instead of scoring a plan against hand-shaped nonlinear curves
(goals.py's old _threshold_term/_stock_bill_term), every goal becomes a
set of ordinary Formula entries that convert a slice of some virtual
*flow* resource (dollar_flow, power_flow, ...) into "prosperity points",
each with strictly-decreasing pp-per-unit slope and its own cap. A plain
$-maximizing LP then naturally fills the steepest-slope tier first, then
the next, and so on -- reproducing a concave, piecewise-linear
diminishing-returns curve with zero nonlinear solver machinery. Three
reusable tier shapes cover every goal:

  - satisfaction_tiers(): a goal anchored at a real 100% target (e.g.
    the Wuling Stock Bill cap, the average power demand, or hitting a
    day's delivery jobs), with two further breakpoints expressed as
    ratios of that target -- soft_cap_ratio (beyond this, marginal
    value drops sharply) and hard_cap_ratio (beyond this, literally
    nothing -- that tier's cap is finite, not infinite). Hard
    Satisfaction Goals (e.g. power -- DIGE's battery-balancer
    convention of ~5% max overshoot) and Soft Satisfaction Goals (e.g.
    sellable goods -- selling down to 0 faster, tolerating longer
    outages) aren't different curve shapes, just different
    (soft_cap_ratio, hard_cap_ratio) choices on the same one.
  - nonzero_production_tiers(): for goals with no real target at all,
    just "some" production being comfortable (e.g. the Gear
    Components) -- front-loaded, absolute-cap-anchored, with an
    unbounded (never-worthless) tail.
  - hard_satisfaction_bonus(): a genuine discrete "reached the goal in
    one go" bonus, via a single integer=True, limit=1 Formula --
    unlike the smooth tiers, this can only be earned all-at-once (no
    fractional credit), a real discontinuity the smooth tiers can't
    express alone.

Delivery jobs specifically use _materialize_delivery_quotas: every solid
resource competes to supply at most one box's worth of "delivery quota",
matching the real depot mechanic more faithfully than a flat rate
threshold on one or two hand-picked materials -- after a delivery job
drains box_capacity from whatever has the most, that same material is
very unlikely to still be #1 for the next job, so covering multiple
jobs/day for real requires diversifying across distinct materials, not
leaning on one indefinitely. Verified empirically: with no explicit
per-material goal at all, the LP finds this diversification on its own
(e.g. splitting Sandleaf production between keeping some raw and
shredding the rest into Sandleaf Powder, so one plant supplies two
distinct quota-eligible materials).

is_pp_bookkeeping_formula() lets factorylib.endfield.goals._plan_complexity
exclude these tier/bonus/quota formulas from the fraction-complexity
penalty: a player never builds a physical belt for "dollar_flow" or
"delivery_quota_flow", so pricing their fractions the way real recipe
flows are priced would be pricing something that isn't a real
throughput concern at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from factorylib.endfield.wuling import (
    GOOD_YIELD,
    POWER_YIELD,
    RESOURCE_BELT_SPEED,
    RESOURCE_NAMES,
    WulingConfig,
    build_formulas,
    full_supply,
)
from factorylib.optimize import Formula

_MINUTES_PER_DAY = 24 * 60

FLOW_NAMES = (
    "dollar_flow",
    "power_flow",
    "component_flow",
    "xiranite_component_flow",
    "cuprium_component_flow",
    "hetonite_component_flow",
    "delivery_quota_flow",
)
ALL_NAMES = RESOURCE_NAMES + list(FLOW_NAMES)
N = len(ALL_NAMES)

# Every *solid* resource (belt_speed 30 -- liquids run at 120 and can't
# sit in a depot at all; the forge/metatransfer bookkeeping dimensions
# aren't in RESOURCE_BELT_SPEED at all, so they're excluded for free)
# is a candidate "put this in a delivery job box" material.
_SOLID_RESOURCE_NAMES = tuple(
    name for name in RESOURCE_NAMES if RESOURCE_BELT_SPEED.get(name) == 30.0
)

# $/multiple for every formula that earns real money directly (see
# wuling.py's module docstring for each recipe's own $/unit price).
DOLLAR_EARNER_OUTPUTS = {
    "sc_sell": 54 * 6,
    "lc_sell": 25 * 6,
    "hp_sell": 48 * 6,
    "hx_sell": 27 * 6,
    "ya": 22 * 6,
    "yc": 16 * 6,
    "jincao_tea": 22 * 6,
    "jincao_drink": 16 * 6,
    "xi_sell": 1.0,
    "cp_sell": 1.0,
}

_POWER_ROUTE_NAMES = (
    "thermal_bank",
    "thermal_bank_sc_valley",
    "thermal_bank_hc_valley",
    "sc_power",
    "lc_power",
)

_COMPONENT_OWN_FLOWS = {
    "xiranite_component": "xiranite_component_flow",
    "cuprium_component": "cuprium_component_flow",
    "hetonite_component": "hetonite_component_flow",
}


def is_pp_bookkeeping_formula(name: str) -> bool:
    """True for a pp-tier/bonus/delivery-quota formula (see
    build_pp_formulas) -- factorylib.endfield.goals._plan_complexity
    excludes these from the fraction-complexity penalty, same reasoning
    as wuling.is_forge_or_metatransfer_formula: they're pure bookkeeping
    constructs, not real physical throughput."""
    return name.startswith("pp_") or name.startswith("delivery_quota_from_")


def _geometric_split(total: float, n: int, alpha: float) -> list[float]:
    """Split `total` into n positive widths, front-loaded: each width is
    alpha fraction of what's left, except the last, which absorbs
    whatever remains exactly -- so the widths always sum to precisely
    `total` regardless of rounding, with no width hand-picked."""
    if n <= 1:
        return [total]
    widths = []
    remaining = total
    for _ in range(n - 1):
        w = alpha * remaining
        widths.append(w)
        remaining -= w
    widths.append(remaining)
    return widths


def _geometric_decay(first: float, n: int, decay: float) -> list[float]:
    """n values starting at `first`, each `decay` fraction of the last
    (front-loaded: earliest tier is worth the most)."""
    return [first * (decay**i) for i in range(n)]


def satisfaction_tiers(
    target: float,
    *,
    soft_cap_ratio: float,
    hard_cap_ratio: float,
    n_ramp_tiers: int = 3,
    first_pp: float = 1000.0,
    pp_decay: float = 0.15,
    cap_alpha: float = 0.6,
    tail_decay: float = 0.05,
) -> list[tuple[float, float]]:
    """(pp_per_unit, cap_width) tiers for a goal with a real 100% target.

    n_ramp_tiers tiers front-load-partition [0, target] (_geometric_split
    /_geometric_decay) -- most of the reward for actually reaching the
    goal. One more tier covers (target, target*soft_cap_ratio],
    continuing the same pp_decay. One final tier covers
    (target*soft_cap_ratio, target*hard_cap_ratio] at tail_decay times
    the previous tier's pp/unit -- a much sharper drop, matching "beyond
    the soft cap, reward diminishes greatly". Nothing exists past
    target*hard_cap_ratio at all: that tier's cap is finite, not
    infinite, so overshooting past it earns literally nothing further.

    Hard Satisfaction Goal (e.g. power): soft_cap_ratio/hard_cap_ratio
    close to 1.0 (e.g. 1.05/1.40 per DIGE's ~5%-overshoot convention).
    Soft Satisfaction Goal (e.g. sellable goods): both ratios much
    larger (e.g. 1.20/3.00) -- overshoot keeps mattering for a long
    stretch before finally being capped.
    """
    ramp_widths = _geometric_split(target, n_ramp_tiers, cap_alpha)
    ramp_pps = _geometric_decay(first_pp, n_ramp_tiers, pp_decay)
    tiers = list(zip(ramp_pps, ramp_widths))

    soft_cap = target * soft_cap_ratio
    hard_cap = target * hard_cap_ratio
    soft_pp = ramp_pps[-1] * pp_decay
    tiers.append((soft_pp, soft_cap - target))
    tail_pp = soft_pp * tail_decay
    tiers.append((tail_pp, hard_cap - soft_cap))
    return tiers


def nonzero_production_tiers(
    *,
    n_tiers: int = 3,
    first_cap: float = 0.1,
    cap_growth: float = 10.0,
    first_pp: float = 500.0,
    pp_decay: float = 0.2,
) -> list[tuple[float, float]]:
    """(pp_per_unit, cap_width) tiers for a goal with no real target at
    all -- just "some" production being comfortable (e.g. Gear
    Components). Unlike satisfaction_tiers, cap widths GROW
    geometrically from an absolute first_cap (there's no target to split
    a range out of); the last tier is unbounded (math.inf), continuing
    the same pp decay -- diminishing, but never worthless, however much
    is produced."""
    widths = [first_cap * (cap_growth**i) for i in range(n_tiers - 1)]
    widths.append(math.inf)
    pps = _geometric_decay(first_pp, n_tiers, pp_decay)
    return list(zip(pps, widths))


def hard_satisfaction_bonus(
    target: float, bonus_pp: float
) -> tuple[float, float, float]:
    """(input_per_multiple, pp, limit) for a genuine discrete "reached
    the goal in one go" bonus: a single integer=True, limit=1 Formula
    that consumes exactly `target` units of flow and pays `bonus_pp` --
    fractional progress (e.g. 0.9 multiples) earns literally nothing
    from this formula (MILP integrality forbids it), unlike
    satisfaction_tiers, which always give proportional credit. The two
    mechanisms coexist and compete for the same flow: the LP decides
    whether the marginal unit of flow is better spent finishing this
    lump bonus or continuing up the smooth tiers."""
    return target, bonus_pp, 1.0


def _materialize_tiers(
    f: dict[str, Formula],
    prefix: str,
    flow_name: str,
    input_per_multiple: float,
    tiers: list[tuple[float, float]],
) -> None:
    """Turn a (pp_per_unit, cap_width) list into named Formula entries,
    each consuming `input_per_multiple` units of flow_name per multiple,
    capped at cap_width / input_per_multiple multiples."""
    for i, (pp_per_unit, cap_width) in enumerate(tiers, start=1):
        vec = np.zeros(N)
        vec[ALL_NAMES.index(flow_name)] = input_per_multiple
        limit = math.inf if math.isinf(cap_width) else cap_width / input_per_multiple
        f[f"{prefix}_{i}"] = Formula(
            consumption=vec, output=pp_per_unit * input_per_multiple, limit=limit
        )


def _materialize_bonus(
    f: dict[str, Formula], name: str, flow_name: str, target: float, bonus_pp: float
) -> None:
    input_per_multiple, pp, limit = hard_satisfaction_bonus(target, bonus_pp)
    vec = np.zeros(N)
    vec[ALL_NAMES.index(flow_name)] = input_per_multiple
    f[name] = Formula(consumption=vec, output=pp, limit=limit, integer=True)


def _materialize_delivery_quotas(
    f: dict[str, Formula],
    box_capacity: float,
    max_multiple_per_material: float,
) -> None:
    """One capped formula per solid resource: `box_capacity` (per day,
    converted to /min) of that material -> 1 unit of delivery_quota_flow,
    capped at max_multiple_per_material (default 1 -- "at most one
    box's worth of quota from any single material"). See module
    docstring for the real-mechanic rationale."""
    input_per_multiple = box_capacity / _MINUTES_PER_DAY
    for resource_name in _SOLID_RESOURCE_NAMES:
        vec = np.zeros(N)
        vec[ALL_NAMES.index(resource_name)] = input_per_multiple
        vec[ALL_NAMES.index("delivery_quota_flow")] = -1.0
        f[f"delivery_quota_from_{resource_name}"] = Formula(
            consumption=vec, output=0.0, limit=max_multiple_per_material
        )


@dataclass
class PPGoals:
    """Every number the pp tier system depends on -- nothing in
    build_pp_formulas should be a bare literal instead of one of these.

    Args:
        dollar_target/power_target: the same real-world targets as the
            old WulingGoals.stock_bill_cap/power_target.
        *_soft_cap_ratio/*_hard_cap_ratio: satisfaction_tiers'
            breakpoints, as ratios of the target -- see its docstring
            and the module docstring's Hard/Soft Satisfaction framing.
        *_bonus_pp: hard_satisfaction_bonus's lump reward for reaching
            that goal's target in one go.
        delivery_box_capacity/delivery_jobs_per_day: the real depot
            mechanic's own numbers (14000/box, 2 jobs/day by default).
        delivery_quota_max_multiple: the "at most one box's worth from
            any single material" cap _materialize_delivery_quotas is
            built around; raise to relax back toward "any solid can
            unlimited-supply quota".
        hetonite_part_first_cap/component_first_cap/own_flow_first_cap:
            nonzero_production_tiers' starting cap for each Nonzero
            Production Goal.
        n_ramp_tiers: how many tiers satisfaction_tiers uses to
            front-load-partition [0, target].
        complexity_weight: weight of the simplicity/denominator penalty
            (factorylib.simplicity.fraction_complexity), same role as
            the old WulingGoals.complexity_weight.
        max_denom: denominator ceiling used when pricing a rate's
            complexity (see factorylib.simplicity.fraction_complexity).
    """

    dollar_target: float = 1090.0
    dollar_soft_cap_ratio: float = 1.20
    dollar_hard_cap_ratio: float = 3.00
    dollar_bonus_pp: float = 2000.0

    power_target: float = 7000.0
    power_soft_cap_ratio: float = 1.05
    power_hard_cap_ratio: float = 1.40
    power_bonus_pp: float = 2000.0

    delivery_box_capacity: float = 14_000.0
    delivery_jobs_per_day: float = 2.0
    delivery_quota_max_multiple: float = 1.0
    delivery_quota_soft_cap_ratio: float = 1.10
    delivery_quota_hard_cap_ratio: float = 2.00
    delivery_quota_bonus_pp: float = 2000.0

    hetonite_part_first_cap: float = 0.1
    component_first_cap: float = 0.1
    own_flow_first_cap: float = 0.1

    n_ramp_tiers: int = 3

    complexity_weight: float = 0.1
    max_denom: int = 1000


def pp_supply(config: WulingConfig) -> np.ndarray:
    """full_supply(config) extended with zeros for the FLOW_NAMES
    dimensions -- a virtual flow (dollar_flow, power_flow, ...) never
    has any external supply of its own, only what real formulas produce
    into it."""
    base = full_supply(config)
    supply = np.zeros(N)
    supply[: len(base)] = base
    return supply


def build_pp_formulas(
    config: WulingConfig, pp_goals: PPGoals | None = None
) -> dict[str, Formula]:
    """Build the full pp-scored formula set: every real recipe formula
    from build_formulas(config), extended to the flow dimensions, plus
    the pp-tier/bonus/delivery-quota formulas for each goal (see module
    docstring).

    $-earning formulas have their $ output zeroed and replaced with
    dollar_flow production; power-route formulas similarly produce
    power_flow instead of nothing. Everything else about the base
    formulas (including forge/metatransfer bookkeeping, and the
    integer flag on any of them) is preserved unchanged.
    """
    pp_goals = pp_goals or PPGoals()
    base = build_formulas(config)

    def extend(vec: np.ndarray) -> np.ndarray:
        out = np.zeros(N)
        out[: len(vec)] = vec
        return out

    f: dict[str, Formula] = {
        name: Formula(
            consumption=extend(formula.consumption),
            output=0.0,
            limit=formula.limit,
            integer=formula.integer,
        )
        for name, formula in base.items()
    }

    for name, dollar_per_multiple in DOLLAR_EARNER_OUTPUTS.items():
        vec = f[name].consumption.copy()
        vec[ALL_NAMES.index("dollar_flow")] = -dollar_per_multiple
        f[name] = Formula(consumption=vec, output=0.0, limit=base[name].limit)

    for name in _POWER_ROUTE_NAMES:
        vec = f[name].consumption.copy()
        vec[ALL_NAMES.index("power_flow")] = -POWER_YIELD[name]
        f[name] = Formula(consumption=vec, output=0.0, limit=base[name].limit)

    # Component formulas draw the real intermediates directly (Hetonite
    # Part, Heavy Xiranite, etc. via their existing consumption) and
    # additionally produce component_flow (the generic Nonzero
    # Production Goal) plus their own dedicated flow (for the per-
    # Component own-flow tiers) -- no separate "hold for crafting"
    # formula needed, since these are new goods with no existing
    # resource to key off.
    for name, own_flow in _COMPONENT_OWN_FLOWS.items():
        vec = extend(base[name].consumption)
        vec[ALL_NAMES.index("component_flow")] = -GOOD_YIELD[name]
        vec[ALL_NAMES.index(own_flow)] = -GOOD_YIELD[name]
        f[name] = Formula(consumption=vec, output=0.0, limit=base[name].limit)
    vec = extend(base["ferrium_component"].consumption)
    vec[ALL_NAMES.index("component_flow")] = -GOOD_YIELD["ferrium_component"]
    f["ferrium_component"] = Formula(
        consumption=vec, output=0.0, limit=base["ferrium_component"].limit
    )

    # ---- Soft Satisfaction Goal: sellable goods ($) ----
    _materialize_tiers(
        f,
        "pp_dollar",
        "dollar_flow",
        pp_goals.dollar_target,
        satisfaction_tiers(
            pp_goals.dollar_target,
            soft_cap_ratio=pp_goals.dollar_soft_cap_ratio,
            hard_cap_ratio=pp_goals.dollar_hard_cap_ratio,
            n_ramp_tiers=pp_goals.n_ramp_tiers,
        ),
    )
    _materialize_bonus(
        f,
        "pp_dollar_bonus",
        "dollar_flow",
        pp_goals.dollar_target,
        pp_goals.dollar_bonus_pp,
    )

    # ---- Hard Satisfaction Goal: power ----
    _materialize_tiers(
        f,
        "pp_power",
        "power_flow",
        pp_goals.power_target,
        satisfaction_tiers(
            pp_goals.power_target,
            soft_cap_ratio=pp_goals.power_soft_cap_ratio,
            hard_cap_ratio=pp_goals.power_hard_cap_ratio,
            n_ramp_tiers=pp_goals.n_ramp_tiers,
        ),
    )
    _materialize_bonus(
        f,
        "pp_power_bonus",
        "power_flow",
        pp_goals.power_target,
        pp_goals.power_bonus_pp,
    )

    # ---- Delivery Job Quota (Soft Satisfaction Goal anchored at
    # jobs/day -- see module docstring) ----
    _materialize_delivery_quotas(
        f, pp_goals.delivery_box_capacity, pp_goals.delivery_quota_max_multiple
    )
    _materialize_tiers(
        f,
        "pp_delivery_quota",
        "delivery_quota_flow",
        1.0,
        satisfaction_tiers(
            pp_goals.delivery_jobs_per_day,
            soft_cap_ratio=pp_goals.delivery_quota_soft_cap_ratio,
            hard_cap_ratio=pp_goals.delivery_quota_hard_cap_ratio,
            n_ramp_tiers=pp_goals.n_ramp_tiers,
        ),
    )
    _materialize_bonus(
        f,
        "pp_delivery_quota_bonus",
        "delivery_quota_flow",
        pp_goals.delivery_jobs_per_day,
        pp_goals.delivery_quota_bonus_pp,
    )

    # ---- Nonzero Production Goal: Hetonite Part + Components ----
    _materialize_tiers(
        f,
        "pp_hetonite_part",
        "hetonite_part",
        1.0,
        nonzero_production_tiers(first_cap=pp_goals.hetonite_part_first_cap),
    )
    _materialize_tiers(
        f,
        "pp_component",
        "component_flow",
        1.0,
        nonzero_production_tiers(first_cap=pp_goals.component_first_cap),
    )
    for own_flow, prefix in [
        ("xiranite_component_flow", "pp_xiranite"),
        ("cuprium_component_flow", "pp_cuprium"),
        ("hetonite_component_flow", "pp_hetonite"),
    ]:
        _materialize_tiers(
            f,
            prefix,
            own_flow,
            1.0,
            nonzero_production_tiers(first_cap=pp_goals.own_flow_first_cap),
        )

    return f
