"""Local search over LP-derived production plans (Part 5).

Starting from an LP-optimal solution (which maximizes raw $ only), search
nearby plans for better *fitness* -- a different, generally nonconvex
objective (see factorylib.endfield.goals.fitness) that also rewards simple
fractions and secondary goals. Five moves:

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
  - shift: reallocate an already-fully-consumed resource between two
    formulas that compete for it (e.g. a battery split between selling
    and power). round_down can't propose this by itself when the donor
    formula's rate is already at its simplest denominator (nothing
    smaller to try), and allocate_slack can't either, since there's no
    *unused* slack to hand out -- the resource is fully claimed by the
    donor already. shift picks a random donor/target pair that both
    positively consume some shared resource, shrinks the donor down to a
    nearby "nice" denominator, and hands everything that frees (plus any
    pre-existing slack) to the target -- also snapped to a nice
    denominator -- bounded by the target's own run-rate limit. See
    _shift_move.
  - pinned_lp: commit a random handful of currently-active formulas'
    rates as lower bounds ("keep at least this many multiples running"),
    then re-solve the raw $-maximization LP for every formula subject to
    those floors. Unlike the other four moves, which each nudge one or
    two formulas at a time, this re-optimizes the *entire* remaining
    allocation in one exact step -- it can reach reallocations that no
    sequence of single-formula nudges would ever stumble into, at the
    cost of the freshly-reoptimized formulas generally landing on
    arbitrary (not "nice") fractions, which the other moves then get a
    chance to clean up on subsequent iterations. See _pinned_lp_move.
  - toggle_integer: flip one Formula.integer=True formula fully on (its
    full `limit` multiples) or fully off (zero). Formula.integer covers
    both a genuine MILP-style integer count (e.g. wuling.py's
    xiranite_forge_alloc, any whole number of forges 0..max_forges) and a
    limit=1 all-or-nothing bonus (e.g.
    factorylib.endfield.pp_goals.hard_satisfaction_bonus's "reached the
    goal in one go" bonuses) -- either way, only whole-number rates are
    ever valid, never e.g. 0.7 multiples. The other five moves all
    operate on plain continuous rates and have no notion of this
    invariant, so _integer_rates_valid rejects any of their proposals
    that would leave an integer formula fractional; toggle_integer (plus
    pinned_lp's MILP re-solve, now that lp_with_floor preserves the
    integer flag) is how the search actually reaches a nonzero
    whole-number rate for one of these instead of just getting stuck
    rejecting fractional proposals forever. See _toggle_integer_move.

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

from factorylib.optimize import Formula, maximize_dollar

_NICE_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)

FitnessFn = Callable[[np.ndarray], float]


MoveFn = Callable[
    [np.ndarray, list[Formula], np.ndarray, np.ndarray, random.Random],
    "np.ndarray | None",
]


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
        extra_moves: additional proposal functions, each
            (rates, formulas, consumption, supply, rng) -> new_rates or
            None, given equal weight alongside the five built-in moves.
            Lets a domain layer (e.g. factorylib.endfield.refine) inject
            moves that need knowledge search.py deliberately doesn't have
            -- e.g. targeting a specific known goal minimum on a
            currently-zero-$-output formula, which no $-only move (like
            pinned_lp) would ever choose on its own. See lp_with_floor
            for the shared mechanic such a move would typically use.
    """

    iterations: int = 2000
    initial_temperature: float = 5.0
    cooling_rate: float = 0.995
    denominators: tuple[int, ...] = _NICE_DENOMINATORS
    seed: int | None = None
    extra_moves: tuple[MoveFn, ...] = ()


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
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    denominators: tuple[int, ...],
    rng: random.Random,
) -> np.ndarray | None:
    """Round a nonzero rate DOWN to a simpler nearby fraction.

    Shrinking a formula always frees the resources it *consumes* (positive
    consumption coefficients), but if it's also a net *producer* of some
    resource (negative coefficient) that a different, already-fixed
    formula depends on, shrinking it can starve that other consumer --
    verify network-wide feasibility rather than assuming a reduction is
    always safe."""
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
    if np.any(consumption @ new_rates > supply + 1e-9):
        return None
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


def _shift_move(
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    denominators: tuple[int, ...],
    rng: random.Random,
) -> np.ndarray | None:
    """Shrink one formula (the donor) and hand whatever that frees to a
    different formula (the target) that competes with it for the same
    resource. Neither round_down (only ever proposes smaller
    denominators of the donor's *own* rate, not a full reallocation) nor
    allocate_slack (only spends slack that's already unused) can express
    this: if a resource is already 100% claimed by one consumer, a target
    formula that also needs it can never be grown by those two moves
    alone, however good the trade would be.

    The donor is always shrunk down to one of the "nice" denominators
    (like round_down), never by an arbitrary continuous amount -- a raw
    random fraction of the donor's rate would almost never land on a
    simple fraction itself, silently reintroducing the same complexity
    this whole search exists to avoid."""
    n = len(formulas)
    if n < 2:
        return None
    j = rng.randrange(n)
    col_j = consumption[:, j]
    shared = [k for k in range(len(col_j)) if col_j[k] > 1e-12]
    if not shared:
        return None
    donors = [
        i
        for i in range(n)
        if i != j and rates[i] > 1e-9 and any(consumption[k, i] > 1e-12 for k in shared)
    ]
    if not donors:
        return None
    i = rng.choice(donors)
    r = float(rates[i])
    d = rng.choice(denominators)
    # Step down to the *previous* 1/d multiple below r, not just floor(r*d)/d
    # -- when r is itself already an exact multiple of 1/d (the common case
    # for a donor sitting at a "nice", already-simple rate, e.g. an
    # integer), floor(r*d)/d == r and would never actually shrink it.
    new_r = math.floor(r * d - 1e-9) / d
    if new_r >= r - 1e-12:
        return None
    delta_i = r - new_r

    trial_rates = rates.copy()
    trial_rates[i] -= delta_i
    # Shrinking the donor is only guaranteed safe for the resource(s) it
    # shares with the target (freed, not tightened). If the donor is also
    # a net producer of some *other* resource (negative coefficient) that
    # a third formula already fully relies on, this reduction can starve
    # that dependency -- same risk _round_down_move guards against.
    if np.any(consumption @ trial_rates > supply + 1e-9):
        return None
    remaining_slack = supply - consumption @ trial_rates
    bounds = [remaining_slack[k] / col_j[k] for k in shared]
    max_delta_j = min(bounds)
    max_delta_j = min(max_delta_j, formulas[j].limit - trial_rates[j])
    if not math.isfinite(max_delta_j) or max_delta_j <= 1e-9:
        return None
    # Snap the target's new rate to a nice denominator too, rather than
    # handing it the exact (generally ugly) freed amount -- same
    # reasoning as the donor above, just rounding up instead of down
    # (mirrors _round_up_move's "largest nice fraction that still fits").
    upper_bound = trial_rates[j] + max_delta_j
    d_j = rng.choice(denominators)
    new_j = math.floor(upper_bound * d_j) / d_j
    if new_j <= trial_rates[j] + 1e-12:
        return None
    trial_rates[j] = new_j
    if np.any(consumption @ trial_rates > supply + 1e-9):
        return None
    return trial_rates


def _integer_rates_valid(
    rates: np.ndarray, formulas: list[Formula], tol: float = 1e-6
) -> bool:
    """True if every Formula.integer=True formula's rate is (within tol) a
    whole number of multiples, in [0, limit] -- Formula.integer covers
    two shapes: genuine MILP-style integer choices with limit > 1 (e.g.
    wuling.py's xiranite_forge_alloc, 0..max_forges forges), where any
    whole count is valid, and hard_satisfaction_bonus-style limit=1
    all-or-nothing bonuses, where 0 or 1 are the only whole counts
    possible anyway -- so a single "is this a whole number" check covers
    both without special-casing the limit. round_down/round_up/
    allocate_slack/shift/pinned_lp all operate on plain continuous rates
    with no notion of this flag, so this is checked centrally on every
    proposal rather than taught to each move individually -- a proposal
    that fails this is rejected outright, same as one that violates
    supply."""
    for rate, f in zip(rates, formulas):
        if not f.integer:
            continue
        if rate < -tol or rate > f.limit + tol:
            return False
        if abs(rate - round(rate)) > tol:
            return False
    return True


def _snap_integer_rates(rates: np.ndarray, formulas: list[Formula]) -> np.ndarray:
    """Round every Formula.integer=True dimension to the nearest whole
    number of multiples, clipped to [0, limit] -- defensive sanitization
    for initial_rates, which may come from a caller that never
    guaranteed the whole-number invariant itself."""
    rates = rates.copy()
    for i, f in enumerate(formulas):
        if not f.integer:
            continue
        snapped = round(rates[i])
        if math.isfinite(f.limit):
            snapped = min(snapped, f.limit)
        rates[i] = max(snapped, 0.0)
    return rates


def _toggle_integer_move(
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    """Flip one Formula.integer=True formula fully on (its full `limit`
    multiples) or fully off (zero) -- see module docstring's
    toggle_integer entry. For a limit=1 all-or-nothing bonus (see
    factorylib.endfield.pp_goals.hard_satisfaction_bonus) this is the
    only move that can turn it on at all, since allocate_slack only
    reaches the full limit when slack happens to cover it exactly and
    round_up/round_down have no smaller-denominator candidate to try
    from 0 or from an already-whole rate. For a larger-limit integer
    formula (e.g. wuling.py's xiranite_forge_alloc), it's a coarse
    on/off jump alongside pinned_lp's finer-grained MILP re-solves.

    Turning the formula fully OFF is not automatically safe: like
    _round_down_move/_shift_move, if it's also a net *producer* of some
    resource (negative consumption coefficient -- e.g.
    heavy_xiranite_forge_alloc producing hx_forge_capacity) that a
    different, already-fixed formula depends on, dropping it to zero can
    starve that dependency -- verify network-wide feasibility rather
    than assuming it's always safe to switch off."""
    candidates = [
        i for i, f in enumerate(formulas) if f.integer and math.isfinite(f.limit)
    ]
    if not candidates:
        return None
    i = rng.choice(candidates)
    new_rates = rates.copy()
    new_rates[i] = 0.0 if rates[i] > 1e-6 else formulas[i].limit
    if np.any(consumption @ new_rates > supply + 1e-9):
        return None
    return new_rates


def lp_with_floor(
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    floor: np.ndarray,
) -> np.ndarray | None:
    """Re-solve the raw $-maximization LP for every formula, each with a
    lower bound of floor[i] ("keep at least this many multiples
    running"), given whatever supply remains after crediting the floors'
    own production/consumption. Returns None if the floor alone would
    already need more of some resource than the rest of the network can
    ever supply -- e.g. a floor on a formula that consumes a purely-
    internal, zero-external-supply resource without also crediting
    whatever produces it.

    Shared by _pinned_lp_move (floors built from the plan's own
    currently-active rates) and by domain-specific moves elsewhere (e.g.
    factorylib.endfield.refine's goal-targeted move, which floors a
    formula at a known goal minimum even when it's currently at zero --
    something no raw-$ re-solve would ever choose on its own, since a
    zero-$-output formula never earns its way into the LP's solution by
    itself)."""
    n = len(formulas)
    shifted_supply = supply - consumption @ floor
    if np.any(shifted_supply < -1e-9):
        return None
    shifted_supply = np.maximum(shifted_supply, 0.0)
    shifted_formulas = [
        Formula(
            consumption=formulas[i].consumption,
            output=formulas[i].output,
            limit=(
                formulas[i].limit - floor[i]
                if math.isfinite(formulas[i].limit)
                else math.inf
            ),
            integer=formulas[i].integer,
        )
        for i in range(n)
    ]
    result = maximize_dollar(shifted_supply, shifted_formulas)
    if result.status != "optimal":
        return None
    new_rates = floor + result.formula_rates
    if np.any(consumption @ new_rates > supply + 1e-6):
        return None
    return new_rates


def _pinned_lp_move(
    rates: np.ndarray,
    formulas: list[Formula],
    consumption: np.ndarray,
    supply: np.ndarray,
    rng: random.Random,
    max_pinned: int = 3,
) -> np.ndarray | None:
    """Commit a random handful of currently-active formulas' rates as
    lower bounds, then re-solve via lp_with_floor. This is conservative,
    not exhaustive: some pinned subsets that could actually work get
    rejected too (see lp_with_floor), which just means this proposal is
    skipped in favor of another on the next iteration."""
    n = len(formulas)
    active = [i for i in range(n) if rates[i] > 1e-9]
    if not active:
        return None
    k = rng.randint(1, min(max_pinned, len(active)))
    pinned = rng.sample(active, k)
    floor = np.zeros(n)
    for i in pinned:
        floor[i] = rates[i]
    return lp_with_floor(formulas, consumption, supply, floor)


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
    current = _snap_integer_rates(initial, formulas)
    current_fitness = fitness_fn(current)
    best, best_fitness = current.copy(), current_fitness

    denominators = config.denominators
    move_fns: list[Callable[[np.ndarray, random.Random], np.ndarray | None]] = [
        lambda r, rg: _round_down_move(
            r, formulas, consumption, supply, denominators, rg
        ),
        lambda r, rg: _round_up_move(
            r, formulas, consumption, supply, denominators, rg
        ),
        lambda r, rg: _allocate_slack_move(r, formulas, consumption, supply, rg),
        lambda r, rg: _shift_move(r, formulas, consumption, supply, denominators, rg),
        lambda r, rg: _pinned_lp_move(r, formulas, consumption, supply, rg),
        lambda r, rg: _toggle_integer_move(r, formulas, consumption, supply, rg),
    ]
    move_fns.extend(
        (lambda mv: lambda r, rg: mv(r, formulas, consumption, supply, rg))(extra)
        for extra in config.extra_moves
    )

    temperature = config.initial_temperature
    accepted = 0
    for _ in range(config.iterations):
        proposal = move_fns[rng.randrange(len(move_fns))](current, rng)

        if proposal is not None and not _integer_rates_valid(proposal, formulas):
            proposal = None

        if proposal is not None:
            proposal_fitness = fitness_fn(proposal)
            delta = proposal_fitness - current_fitness
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
                current, current_fitness = proposal, proposal_fitness
                accepted += 1
                if current_fitness > best_fitness:
                    best, best_fitness = current.copy(), current_fitness

        temperature *= config.cooling_rate

    # Every accepted proposal was already gated by _integer_rates_valid
    # above, so this should never fire -- but it's cheap, and it turns a
    # future regression (e.g. a new move type that forgets the gate)
    # into a loud, immediate failure instead of a silently-wrong plan
    # (see the module docstring's toggle_integer entry for the real bug
    # this class of check would have caught much sooner).
    assert _integer_rates_valid(best, formulas), (
        "simulated_annealing produced a fractional Formula.integer rate "
        "despite _integer_rates_valid gating every move -- this is a bug "
        "in search.py itself, not a proposal that should have been "
        "rejected"
    )

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
    # dual_annealing has no notion of Formula.integer -- snap those
    # dimensions to the all-or-nothing invariant _integer_rates_valid
    # expects (see module docstring's toggle_integer entry) before the
    # feasibility check below, since snapping one up to its full limit
    # can itself push usage over supply.
    rates = _snap_integer_rates(rates, formulas)
    usage = consumption @ rates
    if np.any(usage > supply + 1e-6):
        # Penalty didn't fully enforce feasibility (or integer-snapping
        # did); fall back to the starting point rather than return an
        # invalid plan.
        rates = _snap_integer_rates(np.asarray(initial_rates, dtype=float), formulas)

    assert _integer_rates_valid(rates, formulas), (
        "scipy_dual_annealing produced a fractional Formula.integer rate "
        "despite _snap_integer_rates -- this is a bug in search.py itself"
    )

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
