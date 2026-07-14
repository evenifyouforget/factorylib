"""Local search over LP-derived production plans (Part 5).

Starting from an LP-optimal solution (which maximizes raw $ only), search
nearby plans for better *fitness* -- a different, generally nonconvex
objective (see factorylib.endfield.goals.fitness) that also rewards simple
fractions and secondary goals. Three moves:

  - round_down: replace one formula's rate with a simpler (smaller-
    denominator) fraction no larger than its current value, freeing up
    the resources it no longer consumes.
  - round_up: replace one formula's rate with a simpler (smaller-
    denominator) fraction no smaller than its current value ("virtual
    limits" in factorylib_tmp_physical_factory_construction.md -- e.g.
    rounding up to consume one more whole belt of some input, since the
    extra output is often harmless). Only proposed when the extra
    consumption fits within currently unused slack (see
    _round_up_move) -- this is exactly what's provably safe here: every
    accepted rates vector satisfies consumption @ rates <= supply for
    every tracked resource, including cyclic ones (sewage/effluent are
    just resource dimensions, not special-cased). This does NOT prove
    physical-topology safety (specific splitter wiring, priority-overflow
    routing, depot turn-taking, or backpressure-driven "auto-limiting"
    where a ratio-limited co-reactant caps achieved flow below nominal
    belt capacity) -- none of that is modeled here. See headroom_loss()
    for a practical proxy for one specific risk the physical-construction
    notes raise: a move that fully saturates a resource which used to
    have spare capacity.
  - allocate_slack: given whatever resources are currently unused (either
    from the start, or freed by a round_down move), increase some
    formula's rate -- new or already active -- by as much as the
    remaining slack and its own run-rate limit allow.

The fitness landscape from combining $ output with a denominator/prime
complexity penalty is nonconvex (a simpler nearby fraction can score
higher despite lower raw output), so plain greedy hill-climbing on these
moves easily gets stuck in a bad local optimum. Simulated annealing --
accepting some worse moves early on, with the acceptance probability
shrinking as the "temperature" cools -- gives the search a chance to
cross those dips.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

from factorylib.optimize import Formula

_NICE_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class SearchConfig:
    """Simulated annealing parameters.

    Args:
        iterations: number of proposed moves to try.
        initial_temperature: starting temperature, in fitness-score units;
            controls how readily worse moves are accepted early on.
        cooling_rate: per-iteration temperature multiplier
            (0 < cooling_rate < 1); lower cools faster.
        denominators: candidate "nice" denominators tried by round_down,
            all built from small prime factors (2s, 3s) since those are
            the easiest to physically split -- see factorylib.simplicity.
        seed: RNG seed, for reproducibility.
    """

    iterations: int = 2000
    initial_temperature: float = 5.0
    cooling_rate: float = 0.995
    denominators: tuple[int, ...] = _NICE_DENOMINATORS
    seed: int | None = None


@dataclass
class SearchOutcome:
    """Result of a search() call.

    Attributes:
        rates: the best (highest-fitness) rates vector found.
        fitness: fitness_fn(rates) for that vector.
        accepted_moves: number of proposed moves accepted (diagnostic).
        proposed_moves: number of moves proposed in total (diagnostic).
        headroom_lost: resource indices that had spare capacity under
            initial_rates but are fully saturated under rates -- see
            headroom_loss(). A diagnostic, not a rejection.
    """

    rates: np.ndarray
    fitness: float
    accepted_moves: int
    proposed_moves: int
    headroom_lost: list[int]


def headroom_loss(
    supply: np.ndarray,
    consumption: np.ndarray,
    before_rates: np.ndarray,
    after_rates: np.ndarray,
    tol: float = 1e-9,
) -> list[int]:
    """Resource indices that had spare capacity (slack > tol) under
    before_rates but are fully saturated (slack <= tol) under after_rates.

    A practical proxy for one specific risk
    factorylib_tmp_physical_factory_construction.md raises: "clogging a
    sewage loop that would have had excess capacity otherwise." This is
    diagnostic, not a rejection -- many economically-good moves
    legitimately saturate a resource that has to be fully used to be
    worthwhile; it does not by itself indicate a problem. It's also not a
    physical-topology check (see module docstring): it only reports on
    the linear resource balance this model tracks.
    """
    before_slack = supply - consumption @ before_rates
    after_slack = supply - consumption @ after_rates
    return [
        k for k in range(len(supply)) if before_slack[k] > tol and after_slack[k] <= tol
    ]


def _round_down_move(
    rates: np.ndarray, denominators: tuple[int, ...], rng: random.Random
) -> np.ndarray | None:
    nonzero = [i for i, r in enumerate(rates) if r > 1e-9]
    if not nonzero:
        return None
    i = rng.choice(nonzero)
    r = float(rates[i])
    current_denom = Fraction(r).limit_denominator(1000).denominator
    candidates = [d for d in denominators if d < current_denom]
    if not candidates:
        return None
    d = rng.choice(candidates)
    new_r = math.floor(r * d) / d
    if new_r >= r:
        return None
    new_rates = rates.copy()
    new_rates[i] = new_r
    return new_rates


def _round_up_move(
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    denominators: tuple[int, ...],
    rng: random.Random,
) -> np.ndarray | None:
    """Round a nonzero rate UP to a simpler nearby fraction ("virtual
    limits" -- see module docstring), if the extra consumption fits
    within currently unused slack. Uses the exact same feasibility bound
    as _allocate_slack_move, so this is provably safe with respect to the
    linear resource balance (see headroom_loss() for what it does NOT
    prove)."""
    nonzero = [i for i, r in enumerate(rates) if r > 1e-9]
    if not nonzero:
        return None
    i = rng.choice(nonzero)
    r = float(rates[i])
    current_denom = Fraction(r).limit_denominator(1000).denominator
    candidates = [d for d in denominators if d < current_denom]
    if not candidates:
        return None
    d = rng.choice(candidates)
    new_r = math.ceil(r * d) / d
    if new_r <= r:
        return None
    delta = new_r - r

    if delta > formulas[i].limit - r + 1e-9:
        return None
    remaining_slack = supply - consumption @ rates
    col = consumption[:, i]
    for k in range(len(col)):
        if col[k] > 1e-12 and delta * col[k] > remaining_slack[k] + 1e-9:
            return None

    new_rates = rates.copy()
    new_rates[i] = new_r
    return new_rates


def _allocate_slack_move(
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    j = rng.randrange(len(formulas))
    usage = consumption @ rates
    remaining_slack = supply - usage
    col = consumption[:, j]
    # Resources this formula consumes (col > 0) bound how much more of it
    # can run; resources it produces or ignores (col <= 0) impose no upper
    # bound from this constraint (more production only frees more slack).
    bounds = [remaining_slack[k] / col[k] for k in range(len(col)) if col[k] > 1e-12]
    max_delta = min(bounds) if bounds else math.inf
    max_delta = min(max_delta, formulas[j].limit - rates[j])
    if not math.isfinite(max_delta) or max_delta <= 1e-9:
        return None
    new_rates = rates.copy()
    new_rates[j] += max_delta
    return new_rates


def simulated_annealing(
    supply: np.ndarray,
    formulas: list[Formula],
    initial_rates: np.ndarray,
    fitness_fn: FitnessFn,
    config: SearchConfig | None = None,
) -> SearchOutcome:
    """Simulated-annealing local search starting from initial_rates
    (typically an LP-optimal solution), maximizing fitness_fn(rates)."""
    config = config or SearchConfig()
    rng = random.Random(config.seed)
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.asarray(supply, dtype=float)

    initial = np.asarray(initial_rates, dtype=float).copy()
    current = initial.copy()
    current_fitness = fitness_fn(current)
    best, best_fitness = current.copy(), current_fitness

    temperature = config.initial_temperature
    accepted = 0
    for _ in range(config.iterations):
        move = rng.random()
        if move < 1 / 3:
            proposal = _round_down_move(current, config.denominators, rng)
        elif move < 2 / 3:
            proposal = _round_up_move(
                current, formulas, consumption, supply, config.denominators, rng
            )
        else:
            proposal = _allocate_slack_move(current, formulas, consumption, supply, rng)

        if proposal is not None:
            proposal_fitness = fitness_fn(proposal)
            delta = proposal_fitness - current_fitness
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
                current, current_fitness = proposal, proposal_fitness
                accepted += 1
                if current_fitness > best_fitness:
                    best, best_fitness = current.copy(), current_fitness

        temperature *= config.cooling_rate

    return SearchOutcome(
        rates=best,
        fitness=best_fitness,
        accepted_moves=accepted,
        proposed_moves=config.iterations,
        headroom_lost=headroom_loss(supply, consumption, initial, best),
    )


def scipy_dual_annealing(
    supply: np.ndarray,
    formulas: list[Formula],
    initial_rates: np.ndarray,
    fitness_fn: FitnessFn,
    config: SearchConfig | None = None,
) -> SearchOutcome:
    """Alternative backend: scipy.optimize.dual_annealing over the
    continuous relaxation, with a penalty term for constraint violation.

    Kept for comparison (see factorylib.endfield.refine's module
    docstring for the empirical comparison this rationale is based on): a
    continuous global optimizer has no way to prefer small-denominator
    fractions except through the fitness penalty itself. Worse, if
    initial_rates is a fully resource-saturated LP vertex (zero slack in
    every dimension -- common for these problems), *any* continuous step
    away from it immediately violates some constraint, so the penalty
    drives the search right back to the starting point and it never
    explores at all. The fallback below guards against the rarer case
    where it wanders to a near-feasible point that still violates supply
    once clipped to rates >= 0.
    """
    from scipy.optimize import dual_annealing

    config = config or SearchConfig()
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.asarray(supply, dtype=float)
    bounds = [(0.0, f.limit if math.isfinite(f.limit) else 1e6) for f in formulas]

    def objective(x: np.ndarray) -> float:
        usage = consumption @ x
        violation = np.maximum(usage - supply, 0.0)
        penalty = 1e6 * float(np.sum(violation))
        return -fitness_fn(x) + penalty

    result = dual_annealing(
        objective,
        bounds,
        seed=config.seed,
        maxiter=max(config.iterations // 50, 10),
        x0=np.asarray(initial_rates, dtype=float),
    )
    rates = np.clip(result.x, 0.0, None)
    usage = consumption @ rates
    if np.any(usage > supply + 1e-6):
        # Penalty didn't fully enforce feasibility; fall back to the
        # starting point rather than return an invalid plan.
        rates = np.asarray(initial_rates, dtype=float)

    initial = np.asarray(initial_rates, dtype=float)
    return SearchOutcome(
        rates=rates,
        fitness=fitness_fn(rates),
        accepted_moves=-1,
        proposed_moves=config.iterations,
        headroom_lost=headroom_loss(supply, consumption, initial, rates),
    )


_BACKENDS: dict[str, Callable[..., SearchOutcome]] = {
    "sa": simulated_annealing,
    "scipy": scipy_dual_annealing,
}


def search(
    supply: np.ndarray,
    formulas: list[Formula],
    initial_rates: np.ndarray,
    fitness_fn: FitnessFn,
    config: SearchConfig | None = None,
    *,
    backend: str = "sa",
) -> SearchOutcome:
    """Dispatch to a search backend by name. "sa" (the discrete-move
    simulated annealing above) is the default -- see rationale in
    factorylib.endfield.refine."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choices: {sorted(_BACKENDS)}")
    return _BACKENDS[backend](supply, formulas, initial_rates, fitness_fn, config)
