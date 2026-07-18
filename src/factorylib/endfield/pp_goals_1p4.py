"""Prosperity Points (pp) for Endfield 1.4 -- same system as pp_goals.py
(see its module docstring for the full design rationale: satisfaction_tiers/
nonzero_production_tiers/hard_satisfaction_bonus are generic, resource-
agnostic pure functions, reused here unchanged), wired to wuling_1p4's
extended resource set instead of 1.2e's.

New Nonzero Production Goals, per tmp_notes/new_goals.md's priority list
(all modeled as nonzero_production_tiers -- "some" production being
comfortable, not a real 100% target -- just with different first_cap
magnitudes reflecting the note's own Low/Mid/High priority framing):
  - Low priority (first_cap=0.1): Separator Core, Cuprium Canister, and
    both filled variants (Inergen/Xiragen).
  - Mid priority (first_cap=0.5): Liquid Heavy Xiranite, Pyrrolite Part,
    the T1-T4 Crafting Point chain (see below), and -- raised from 1.2e's
    own 0.1 default -- Cuprium Component and Hetonite Part.
  - High priority (first_cap=10.0): Liquid Xiranite (needed in bulk --
    new_goals.md notes 5000+ for Rift Neutralization).

The T1-T4 Crafting Point chain (flexible_gear_crafting.md) collapses to a
single Nonzero Production Goal keyed on t1_crafting_point alone: every
tier cascades down to T1 for free (t4_to_t3/t3_to_t2/t2_to_t1 are
zero-cost, unconstrained conversions -- see wuling_1p4.build_formulas),
so a plan that produces any Crafting Point at all has no reason not to
cascade it all the way down once rewarded there, and new_goals.md itself
frames "T1~T4 Gear Crafting Point" as one combined line item rather than
four independently-prioritized ones.

Resolves wuling_1p4's own open design question (see tmp_notes/wip_todo.md):
Xiranite/Cuprium/Hetonite Component's own Nonzero Production tiers key
DIRECTLY on their real tracked resource (xiranite_component_item/
cuprium_component_item/hetonite_component_item -- see wuling_1p4's
_component_item_yield) instead of a dedicated per-component *_flow
bookkeeping dimension the way 1.2e's pp_goals.py needed to (1.2e's
xiranite_component/etc. formulas have $0 output and no tracked resource
of their own at all, so a flow was the only way to give them one).
Pyrrolite Component needs no such treatment either -- gearing_unit
already produces it as a genuine tracked resource from the start. Only
the AGGREGATE "any component nonzero" goal (pp_goals.py's own
component_flow) still needs a flow dimension, since no single real
resource represents "any of these five".

Planting Units draw real power (tmp_notes/power_consumption.md), and
Water/Acid are no longer free either (tmp_notes/old_prompt.md's Fluid
Pump ratios) -- both modeled as power_flow consumption, on every formula
in wuling_1p4.FORMULA_WATTS (its own comment has the full derivation and
every ratio's source). Confirmed with the user: this makes the power
goal implicitly grow with how much of this a plan actually does (more
buildings -> more of the same battery output now has to cover their
upkeep before it counts toward the target). Not every building's power
draw is modeled this way (deliberately, per power_consumption.md's own
scope note) -- only the ones that were otherwise completely free of any
resource cost in this model. The SAME FORMULA_WATTS table also drives a
real (if modest) $ tax in wuling_1p4.build_formulas() itself, for the
raw $-maximizing search() -- see its own module comment for why that's a
separate mechanism (a different objective entirely) from this one, even
though they share the same underlying Watt figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from factorylib.endfield import wuling_1p4 as v1p4
from factorylib.endfield.pp_goals import (
    hard_satisfaction_bonus,  # noqa: F401 (re-exported for symmetry with pp_goals)
    is_pp_bookkeeping_formula,  # noqa: F401 (re-exported -- same "pp_"/
    # "delivery_quota_from_" naming convention, unchanged)
    nonzero_production_tiers,
    satisfaction_tiers,
)
from factorylib.optimize import Formula

_MINUTES_PER_DAY = 24 * 60

FLOW_NAMES = (
    "dollar_flow",
    "power_flow",
    "component_flow",
    "delivery_quota_flow",
)
ALL_NAMES = v1p4.RESOURCE_NAMES + list(FLOW_NAMES)
N = len(ALL_NAMES)

# Every *solid* resource (belt_speed 30) is a candidate "put this in a
# delivery job box" material -- same reasoning as pp_goals.py's own
# _SOLID_RESOURCE_NAMES, just against wuling_1p4's extended resource set
# (gases are pipe-speed 120, correctly excluded, same as liquids).
_SOLID_RESOURCE_NAMES = tuple(
    name for name in v1p4.RESOURCE_NAMES if v1p4.RESOURCE_BELT_SPEED.get(name) == 30.0
)

# $/multiple for every formula that earns real money directly. Canonical
# home is wuling_1p4.py (its own build_formulas needs the same table, to
# reroute each price through raw_dollar for the power/Water/Acid $ tax --
# see its module comment) -- re-exported here unchanged so existing
# importers of pp_goals_1p4.DOLLAR_EARNER_OUTPUTS keep working.
DOLLAR_EARNER_OUTPUTS = v1p4.DOLLAR_EARNER_OUTPUTS

# No new power routes in 1.4 -- reused unchanged.
_POWER_ROUTE_NAMES = (
    "thermal_bank",
    "thermal_bank_sc_valley",
    "thermal_bank_hc_valley",
    "sc_power",
    "lc_power",
)

# Every formula that draws real power (or implicitly needs Water/Acid,
# converted to an equivalent Watt cost -- see wuling_1p4.FORMULA_WATTS'
# own comment for the full derivation and every ratio's source) consumes
# power_flow here too -- the SAME shared flow the Hard Satisfaction power
# tiers below draw from, so "the power goal" isn't a fixed target:
# growing more of these buildings genuinely eats into the same battery
# output that would otherwise count toward it, without needing any
# change to satisfaction_tiers/PPGoals1p4.power_target itself. Reuses
# wuling_1p4's own table rather than a separate, narrower copy, so both
# the raw-$-tax mechanism and this real 7000W-goal accounting stay
# consistent about which buildings draw power and by how much.
_FORMULA_WATTS = v1p4.FORMULA_WATTS

# Every formula contributing to the AGGREGATE "any component nonzero"
# goal (component_flow), and how many items of ITS OWN component it
# produces per multiple (same numbers as wuling_1p4.GOOD_YIELD/
# v1p2e.GOOD_YIELD -- duplicated here rather than imported so this table
# reads as a complete, self-contained list of contributors, matching
# pp_goals.py's own _COMPONENT_OWN_FLOWS convention).
_COMPONENT_FLOW_CONTRIBUTORS = {
    "ferrium_component": 6.0,
    "xiranite_component": 6.0,
    "cuprium_component": 6.0,
    "hetonite_component": 6.0,
    "gearing_unit": 1.0,
}


def _materialize_tiers(
    f: dict[str, Formula],
    prefix: str,
    flow_name: str,
    input_per_multiple: float,
    tiers: list[tuple[float, float]],
) -> None:
    """Same shape as pp_goals._materialize_tiers, against this module's
    own N/ALL_NAMES."""
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
    input_per_multiple = box_capacity / _MINUTES_PER_DAY
    for resource_name in _SOLID_RESOURCE_NAMES:
        vec = np.zeros(N)
        vec[ALL_NAMES.index(resource_name)] = input_per_multiple
        vec[ALL_NAMES.index("delivery_quota_flow")] = -1.0
        f[f"delivery_quota_from_{resource_name}"] = Formula(
            consumption=vec, output=0.0, limit=max_multiple_per_material
        )


@dataclass
class PPGoals1p4:
    """Every number the 1.4 pp tier system depends on -- see PPGoals
    (pp_goals.py) for the dollar/power/delivery fields' own rationale,
    unchanged here. New fields cover new_goals.md's priority list (see
    module docstring) plus two 1.2e defaults raised per that same list.
    """

    dollar_target: float = 1090.0
    dollar_soft_cap_ratio: float = 1.20
    dollar_hard_cap_ratio: float = 3.00
    dollar_bonus_pp: float = 2000.0

    power_target: float = 7000.0
    power_soft_cap_ratio: float = 1.05
    power_hard_cap_ratio: float = 1.10
    power_bonus_pp: float = 2000.0

    delivery_box_capacity: float = 12_000.0
    delivery_jobs_per_day: float = 2.0
    delivery_quota_max_multiple: float = 1.0
    delivery_quota_soft_cap_ratio: float = 1.10
    delivery_quota_hard_cap_ratio: float = 2.00
    delivery_quota_bonus_pp: float = 2000.0

    # Nonzero Production Goals carried over from 1.2e.
    hetonite_part_first_cap: float = 0.5  # raised from 1.2e's 0.1 (new_goals.md: Mid)
    component_first_cap: float = 0.1  # aggregate "any component" -- unchanged
    xiranite_component_first_cap: float = 0.1
    cuprium_component_first_cap: float = 0.5  # raised from 0.1 (new_goals.md: Mid)
    hetonite_component_first_cap: float = 0.1

    # New Nonzero Production Goals (new_goals.md priority list).
    pyrrolite_component_first_cap: float = 0.1
    separator_core_first_cap: float = 0.1
    cuprium_canister_first_cap: float = 0.1
    cuprium_canister_inergen_first_cap: float = 0.1
    cuprium_canister_xiragen_first_cap: float = 0.1
    liquid_heavy_xiranite_first_cap: float = 0.5
    pyrrolite_part_first_cap: float = 0.5
    crafting_point_first_cap: float = 0.5
    liquid_xiranite_first_cap: float = 10.0

    n_ramp_tiers: int = 3

    complexity_weight: float = 0.1
    max_denom: int = 1000


def pp_supply(config: v1p4.WulingConfig1p4) -> np.ndarray:
    """v1p4.full_supply(config) extended with zeros for the FLOW_NAMES
    dimensions -- same role as pp_goals.pp_supply."""
    base = v1p4.full_supply(config)
    supply = np.zeros(N)
    supply[: len(base)] = base
    return supply


def build_pp_formulas(
    config: v1p4.WulingConfig1p4, pp_goals: PPGoals1p4 | None = None
) -> dict[str, Formula]:
    """1.4 equivalent of pp_goals.build_pp_formulas: every real recipe
    formula from wuling_1p4.build_formulas(config), extended to the flow
    dimensions, plus the pp-tier/bonus/delivery-quota formulas for each
    goal (see module docstring for what's new vs. 1.2e)."""
    pp_goals = pp_goals or PPGoals1p4()
    base = v1p4.build_formulas(config)

    def extend(vec: np.ndarray) -> np.ndarray:
        out = np.zeros(N)
        out[: len(vec)] = vec
        return out

    # Note: base's formulas may carry a negative Formula.output already
    # (see wuling_1p4.WulingConfig1p4.power_dollar_tax) -- irrelevant
    # here, since every formula below gets output=0.0 unconditionally
    # (this module scores plans by pp_output, not by summing .output at
    # all) and FORMULA_WATTS' own power_flow tax further down is applied
    # fresh, independent of whatever wuling_1p4.build_formulas() did.
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
        f[name] = Formula(
            consumption=vec,
            output=0.0,
            limit=base[name].limit,
            integer=base[name].integer,
        )

    for name in _POWER_ROUTE_NAMES:
        vec = f[name].consumption.copy()
        vec[ALL_NAMES.index("power_flow")] = -v1p4.POWER_YIELD[name]
        f[name] = Formula(
            consumption=vec,
            output=0.0,
            limit=base[name].limit,
            integer=base[name].integer,
        )

    # Every power/Water/Acid-drawing formula (wuling_1p4.FORMULA_WATTS)
    # draws real power_flow here too -- see this module's own top-level
    # comment for why (competing with the same pool the power
    # satisfaction tiers below draw from, instead of an adjustment to
    # PPGoals1p4.power_target itself).
    for name, watts_per_multiple in _FORMULA_WATTS.items():
        vec = f[name].consumption.copy()
        vec[ALL_NAMES.index("power_flow")] += watts_per_multiple
        f[name] = Formula(
            consumption=vec,
            output=0.0,
            limit=base[name].limit,
            integer=base[name].integer,
        )

    # Aggregate "any component nonzero" goal -- see module docstring for
    # why the per-component tiers below key directly on the real
    # xiranite_component_item/cuprium_component_item/
    # hetonite_component_item/pyrrolite_component resources instead of
    # needing their own flow dimension the way this aggregate does.
    for name, good_yield in _COMPONENT_FLOW_CONTRIBUTORS.items():
        vec = f[name].consumption.copy()
        vec[ALL_NAMES.index("component_flow")] = -good_yield
        f[name] = Formula(
            consumption=vec,
            output=0.0,
            limit=base[name].limit,
            integer=base[name].integer,
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

    # ---- Delivery Job Quota ----
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

    # ---- Nonzero Production Goals: Hetonite Part + Components ----
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
    for resource_name, prefix, first_cap in [
        (
            "xiranite_component_item",
            "pp_xiranite",
            pp_goals.xiranite_component_first_cap,
        ),
        ("cuprium_component_item", "pp_cuprium", pp_goals.cuprium_component_first_cap),
        (
            "hetonite_component_item",
            "pp_hetonite",
            pp_goals.hetonite_component_first_cap,
        ),
        ("pyrrolite_component", "pp_pyrrolite", pp_goals.pyrrolite_component_first_cap),
    ]:
        _materialize_tiers(
            f,
            prefix,
            resource_name,
            1.0,
            nonzero_production_tiers(first_cap=first_cap),
        )

    # ---- New Nonzero Production Goals (new_goals.md priority list) ----
    for resource_name, prefix, first_cap in [
        ("separator_core", "pp_separator_core", pp_goals.separator_core_first_cap),
        (
            "cuprium_canister",
            "pp_cuprium_canister",
            pp_goals.cuprium_canister_first_cap,
        ),
        (
            "cuprium_canister_inergen",
            "pp_cuprium_canister_inergen",
            pp_goals.cuprium_canister_inergen_first_cap,
        ),
        (
            "cuprium_canister_xiragen",
            "pp_cuprium_canister_xiragen",
            pp_goals.cuprium_canister_xiragen_first_cap,
        ),
        (
            "liquid_heavy_xiranite",
            "pp_liquid_heavy_xiranite",
            pp_goals.liquid_heavy_xiranite_first_cap,
        ),
        ("pyrrolite_part", "pp_pyrrolite_part", pp_goals.pyrrolite_part_first_cap),
        ("t1_crafting_point", "pp_crafting_point", pp_goals.crafting_point_first_cap),
        ("liquid_xiranite", "pp_liquid_xiranite", pp_goals.liquid_xiranite_first_cap),
    ]:
        _materialize_tiers(
            f,
            prefix,
            resource_name,
            1.0,
            nonzero_production_tiers(first_cap=first_cap),
        )

    return f
