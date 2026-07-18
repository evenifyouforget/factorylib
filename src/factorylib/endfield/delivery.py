"""Wire factorylib.delivery's simulator to a Wuling production plan:
which materials would actually pile up, unconsumed, in the depot, and
which one the delivery job's auto-select would pick over time.

Two kinds of accumulator:
  - Base *solid* resources with leftover slack (produced but not consumed
    by any formula in the plan) -- exactly what resource_slack already
    tracks. Liquids (Sewage, Xircon Effluent, Inert Xircon Effluent --
    see RESOURCE_BELT_SPEED) are excluded: the depot cannot store
    liquids at all. Excess liquid supply doesn't accumulate anywhere --
    it just causes backpressure (the producing formula's actual
    achieved rate would throttle to match demand, something this
    steady-state LP doesn't model). Xircon Effluent specifically has no
    disposal at all; Inert Xircon Effluent can be discarded like Sewage,
    but that's normally avoided in favor of purifying it back into more
    Xircon Effluent.
  - The secondary-goal goods that have no consumer at all in this model
    (the four Gear Components -- see factorylib.endfield.wuling's module
    docstring): their entire production rate accumulates, since nothing
    else uses them. thermal_bank is excluded: it produces W (power), not
    a stashable material. sandleaf_powder is no longer in this bucket --
    it's now a tracked resource dimension consumed by ori_to_dop, so its
    *net* surplus (production minus what ori_to_dop actually uses) comes
    through resource_slack instead, same as any other solid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from factorylib.delivery import (
    DeliverySimConfig,
    DeliverySimResult,
    simulate_delivery_selections,
)
from factorylib.endfield.wuling import (
    FORMULA_LABELS,
    GOOD_YIELD,
    RESOURCE_BELT_SPEED,
    RESOURCE_LABELS,
    RESOURCE_NAMES,
)

_SOLID_BELT_SPEED = 30.0

# Secondary-goal formulas producing a stashable material with no
# consumer in this model (thermal_bank produces W, not a material;
# sandleaf_powder now has a real consumer, ori_to_dop, so its surplus
# comes through resource_slack instead -- see module docstring).
_STASHABLE_GOOD_FORMULAS = (
    "ferrium_component",
    "xiranite_component",
    "cuprium_component",
    "hetonite_component",
)


def accumulation_rates(
    rates_by_name: dict[str, float],
    resource_slack: np.ndarray,
    *,
    resource_names: Sequence[str] = RESOURCE_NAMES,
    resource_belt_speed: dict[str, float] = RESOURCE_BELT_SPEED,
    resource_labels: dict[str, str] = RESOURCE_LABELS,
    good_yield: dict[str, float] = GOOD_YIELD,
    formula_labels: dict[str, str] = FORMULA_LABELS,
    stashable_good_formulas: Sequence[str] = _STASHABLE_GOOD_FORMULAS,
) -> dict[str, float]:
    """Every material that would plausibly pile up, unconsumed, in the
    depot, keyed by full label: base *solid* resources with leftover
    slack, plus any of the stashable secondary-goal goods (see module
    docstring). Liquids are never candidates -- the depot can't store
    them. Neither are the virtual bookkeeping dimensions (forge_budget,
    hx_forge_capacity, metatransfer_allowance -- see wuling.py's MILP
    forge/metatransfer unification): they have no entry in
    RESOURCE_BELT_SPEED at all, so excluding only liquids (belt_speed ==
    120) let them slip through as if they were real depot-storable
    goods; requiring belt_speed == 30 (solids) excludes them for free.

    resource_names/resource_belt_speed/resource_labels/good_yield/
    formula_labels/stashable_good_formulas default to 1.2e's own (from
    wuling.py) -- pass wuling_1p4's equivalents for a 1.4 plan instead.
    1.4-specific note: only ferrium_component has NO real consumer
    there (unlike 1.2e, where all four Gear Components are pure dead
    ends) -- Xiranite/Cuprium/Hetonite Component now feed the Crafting
    Point chain via their own real xiranite_component_item/etc.
    resource, so their surplus should come through resource_slack
    (belt_speed=30 there too), not stashable_good_formulas' gross-rate
    path, matching how sandleaf_powder was already fixed once IT gained
    a real consumer (see module docstring)."""
    rates: dict[str, float] = {
        resource_labels.get(name, name): float(slack)
        for name, slack in zip(resource_names, resource_slack)
        if slack > 1e-9 and resource_belt_speed.get(name) == _SOLID_BELT_SPEED
    }
    for name in stashable_good_formulas:
        rate = rates_by_name.get(name, 0.0) * good_yield.get(name, 1.0)
        if rate > 1e-9:
            rates[formula_labels.get(name, name)] = rate
    return rates


def predict_delivery_selections(
    rates_by_name: dict[str, float],
    resource_slack: np.ndarray,
    config: DeliverySimConfig | None = None,
) -> DeliverySimResult:
    """Convenience: accumulation_rates() + simulate_delivery_selections()."""
    return simulate_delivery_selections(
        accumulation_rates(rates_by_name, resource_slack), config
    )
