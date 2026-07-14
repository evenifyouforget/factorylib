"""Predict which material a delivery job's "quick top up" (auto-select
whatever has the highest amount in the depot) would choose over time.

Knowing a good's production *rate* meets some target isn't the same as
knowing it will actually be the one delivery jobs draw from: any other
material sitting in the depot with a higher accumulated amount gets
picked instead. Rather than reason about this analytically, simulate the
depot directly: start empty, let it accumulate for a "startup" period,
then repeatedly run delivery jobs (always taking the single
highest-quantity material) interleaved with a day's worth of further
accumulation, and tally which material got picked each time.
"""

from __future__ import annotations

from dataclasses import dataclass

_MINUTES_PER_DAY = 24 * 60


@dataclass
class DeliverySimConfig:
    """Args:
    startup_days: days of accumulation before the first delivery job
        (the depot starts empty; this models not doing deliveries at
        minute 0).
    simulation_days: number of days to simulate.
    jobs_per_day: delivery jobs run per day.
    box_capacity: items removed from the selected material per job.
    """

    startup_days: float = 1.0
    simulation_days: int = 100
    jobs_per_day: int = 2
    box_capacity: float = 14000.0


def simulate_delivery_selections(
    accumulation_rates: dict[str, float], config: DeliverySimConfig | None = None
) -> dict[str, int]:
    """Simulate the depot's "highest amount" auto-select behavior.

    Args:
        accumulation_rates: name -> net items/min accumulating,
            unconsumed, in the depot (e.g. resource slack, or a good with
            no consumer). Names with a non-positive rate never accumulate
            and are dropped -- they can never be picked.
        config: simulation parameters (see DeliverySimConfig).

    Returns:
        name -> number of times it was selected across the whole
        simulation, for every name with a positive accumulation rate
        (0 if it was never picked).
    """
    config = config or DeliverySimConfig()
    daily_rates = {
        name: rate * _MINUTES_PER_DAY
        for name, rate in accumulation_rates.items()
        if rate > 1e-9
    }
    if not daily_rates:
        return {}

    depot = {name: rate * config.startup_days for name, rate in daily_rates.items()}
    tally = dict.fromkeys(daily_rates, 0)

    for _ in range(config.simulation_days):
        for _ in range(config.jobs_per_day):
            selected = max(depot, key=depot.get)
            if depot[selected] <= 0:
                continue
            depot[selected] -= config.box_capacity
            tally[selected] += 1
        for name in depot:
            depot[name] += daily_rates[name]

    return tally
