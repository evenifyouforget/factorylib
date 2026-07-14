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

import numpy as np

from factorylib.delivery import DeliverySimConfig, simulate_delivery_selections
from factorylib.endfield.wuling import (
    FORMULA_LABELS,
    GOOD_YIELD,
    RESOURCE_BELT_SPEED,
    RESOURCE_LABELS,
    RESOURCE_NAMES,
)

_LIQUID_BELT_SPEED = 120.0

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
    rates_by_name: dict[str, float], resource_slack: np.ndarray
) -> dict[str, float]:
    """Every material that would plausibly pile up, unconsumed, in the
    depot, keyed by full label: base *solid* resources with leftover
    slack, plus any of the stashable secondary-goal goods (see module
    docstring). Liquids are never candidates -- the depot can't store
    them."""
    rates: dict[str, float] = {
        RESOURCE_LABELS.get(name, name): float(slack)
        for name, slack in zip(RESOURCE_NAMES, resource_slack)
        if slack > 1e-9 and RESOURCE_BELT_SPEED.get(name) != _LIQUID_BELT_SPEED
    }
    for name in _STASHABLE_GOOD_FORMULAS:
        rate = rates_by_name.get(name, 0.0) * GOOD_YIELD.get(name, 1.0)
        if rate > 1e-9:
            rates[FORMULA_LABELS.get(name, name)] = rate
    return rates


def predict_delivery_selections(
    rates_by_name: dict[str, float],
    resource_slack: np.ndarray,
    config: DeliverySimConfig | None = None,
) -> dict[str, int]:
    """Convenience: accumulation_rates() + simulate_delivery_selections()."""
    return simulate_delivery_selections(
        accumulation_rates(rates_by_name, resource_slack), config
    )
