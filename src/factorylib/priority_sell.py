"""Greedy priority allocation of a shared budget across named amounts.

Models a game mechanic distinct from factorylib.delivery's depot-select
simulator: an outpost's $ savings only regenerate at a fixed rate (the
"stock bill cap"), so at any instant only that much $ worth of produced
goods can actually be sold. Which goods get sold isn't proportional to
how much of each is produced -- players sell in a fixed priority order,
and whatever doesn't fit in the budget simply accumulates unsold instead
of being sold at a discount or lost.
"""

from __future__ import annotations


def allocate_by_priority(
    amounts: dict[str, float], priority: list[str], cap: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Fill `cap` greedily in `priority` order, then any remaining names
    (in their original dict order). Returns (sold, unsold), each keyed by
    every name in `amounts` with sold[name] + unsold[name] == amounts[name].
    """
    remaining = max(cap, 0.0)
    ordered = list(priority) + [name for name in amounts if name not in priority]
    sold: dict[str, float] = {}
    unsold: dict[str, float] = {}
    for name in ordered:
        amount = amounts.get(name, 0.0)
        take = min(amount, remaining)
        sold[name] = take
        unsold[name] = amount - take
        remaining -= take
    return sold, unsold
