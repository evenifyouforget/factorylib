import math
from fractions import Fraction

import numpy as np

from factorylib.optimize import Formula
from factorylib.search import (
    SearchConfig,
    _allocate_slack_move,
    _integer_rates_valid,
    _pinned_lp_move,
    _round_down_move,
    _round_up_move,
    _shift_move,
    _snap_integer_rates,
    _toggle_integer_move,
    headroom_loss,
    scipy_dual_annealing,
    search,
    simulated_annealing,
)


def test_round_down_move_reduces_rate_and_simplifies_denominator():
    formulas = [
        Formula(consumption=np.array([1.0, 0.0]), output=1.0),
        Formula(consumption=np.array([0.0, 1.0]), output=1.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0, 1.0])
    rng = __import__("random").Random(0)
    rates = np.array([19 / 96, 0.0])
    new_rates = _round_down_move(
        rates, formulas, consumption, supply, (1, 2, 3, 4, 6, 8, 12, 24, 48), rng
    )
    assert new_rates is not None
    assert new_rates[0] <= rates[0]
    new_denom = Fraction(new_rates[0]).limit_denominator(1000).denominator
    assert new_denom < 96


def test_round_down_move_returns_none_when_all_rates_zero():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rng = __import__("random").Random(0)
    rates = np.array([0.0])
    assert (
        _round_down_move(rates, formulas, consumption, supply, (1, 2, 3), rng) is None
    )


def test_round_down_move_rejects_move_that_starves_a_downstream_consumer():
    # formula 0 consumes resource X; formula 1 produces exactly what
    # formula 0 needs (net external supply of X is 0). Shrinking formula
    # 1's rate would drop production below formula 0's fixed consumption
    # -- round_down must reject this, not just assume shrinking any rate
    # is always safe (true only for the resources *it* consumes, not for
    # resources it produces that something else relies on).
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0),
        Formula(consumption=np.array([-1.0]), output=1.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([0.0])
    rates = np.array([19 / 96, 19 / 96])
    rng = __import__("random").Random(0)  # picks index 1 (the producer) first
    assert (
        _round_down_move(
            rates, formulas, consumption, supply, (1, 2, 3, 4, 6, 8, 12, 24, 48), rng
        )
        is None
    )


def test_round_up_move_increases_rate_and_simplifies_denominator():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rates = np.array([19 / 96])
    rng = __import__("random").Random(0)
    new_rates = _round_up_move(
        rates, formulas, consumption, supply, (1, 2, 3, 4, 6, 8, 12, 24, 48), rng
    )
    assert new_rates is not None
    assert new_rates[0] >= rates[0]
    new_denom = Fraction(new_rates[0]).limit_denominator(1000).denominator
    assert new_denom < 96


def test_round_up_move_respects_resource_slack():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])  # already fully saturated at rate=1.0
    rates = np.array([1.0])
    rng = __import__("random").Random(0)
    assert _round_up_move(rates, formulas, consumption, supply, (1, 2, 3), rng) is None


def test_round_up_move_respects_formula_limit():
    formulas = [Formula(consumption=np.array([0.0]), output=1.0, limit=0.3)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([100.0])
    rates = np.array([0.3])
    rng = __import__("random").Random(0)
    assert (
        _round_up_move(rates, formulas, consumption, supply, (1, 2, 3, 4), rng) is None
    )


def test_headroom_loss_flags_newly_saturated_resources():
    supply = np.array([10.0, 10.0])
    consumption = np.array([[1.0, 0.0], [0.0, 1.0]])
    before = np.array([5.0, 5.0])  # both resources have slack
    after = np.array([10.0, 5.0])  # resource 0 now fully saturated
    assert headroom_loss(supply, consumption, before, after) == [0]


def test_headroom_loss_empty_when_nothing_newly_saturated():
    supply = np.array([10.0])
    consumption = np.array([[1.0]])
    before = np.array([10.0])  # already saturated before
    after = np.array([10.0])
    assert headroom_loss(supply, consumption, before, after) == []


def test_allocate_slack_move_respects_resource_constraint():
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0),
        Formula(consumption=np.array([1.0]), output=1.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([10.0])
    rates = np.array([3.0, 0.0])
    rng = __import__("random").Random(0)
    new_rates = _allocate_slack_move(rates, formulas, consumption, supply, rng)
    assert new_rates is not None
    usage = consumption @ new_rates
    assert np.all(usage <= supply + 1e-9)


def test_allocate_slack_move_respects_formula_limit():
    formulas = [Formula(consumption=np.array([0.0]), output=1.0, limit=2.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([100.0])
    rates = np.array([0.0])
    rng = __import__("random").Random(0)
    new_rates = _allocate_slack_move(rates, formulas, consumption, supply, rng)
    assert new_rates is not None
    assert new_rates[0] == 2.0


def test_shift_move_reallocates_a_fully_consumed_resource():
    # formula 0 produces resource X (sc_battery-like); formula 1 already
    # fully consumes it (zero slack); formula 2 also wants X but is
    # currently idle. Neither round_down (formula 1's rate is already at
    # its simplest denominator, denom=1) nor allocate_slack (there's no
    # unused slack to hand out) can ever grow formula 2 -- shift is the
    # only move that can shrink formula 1 to free capacity for it.
    formulas = [
        Formula(consumption=np.array([-6.0]), output=0.0),
        Formula(consumption=np.array([6.0]), output=324.0),
        Formula(consumption=np.array([1.5]), output=0.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([0.0])
    rates = np.array([2.0, 2.0, 0.0])

    denominators = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    found = False
    for seed in range(50):
        rng = __import__("random").Random(seed)
        proposal = _shift_move(rates, formulas, consumption, supply, denominators, rng)
        if proposal is not None and proposal[2] > rates[2] + 1e-9:
            found = True
            assert proposal[1] < rates[1]  # donor (the sell formula) shrank
            new_denom = Fraction(float(proposal[1])).limit_denominator(1000).denominator
            assert new_denom in denominators  # donor lands on a nice fraction too
            usage = consumption @ proposal
            assert np.all(usage <= supply + 1e-9)  # still feasible
            break
    assert found, "shift move never reallocated the shared resource across 50 seeds"


def test_shift_move_returns_none_with_a_single_formula():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rates = np.array([1.0])
    rng = __import__("random").Random(0)
    assert _shift_move(rates, formulas, consumption, supply, (1, 2, 3), rng) is None


def test_shift_move_returns_none_when_no_formula_shares_a_resource():
    formulas = [
        Formula(consumption=np.array([1.0, 0.0]), output=1.0),
        Formula(consumption=np.array([0.0, 1.0]), output=1.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0, 1.0])
    rates = np.array([1.0, 1.0])
    for seed in range(20):
        rng = __import__("random").Random(seed)
        assert _shift_move(rates, formulas, consumption, supply, (1, 2, 3), rng) is None


def test_pinned_lp_move_reoptimizes_the_unpinned_remainder():
    # formula 0 and 1 both have low value/unit and split the entire
    # supply between them; formula 2 (higher value/unit) is idle. No
    # single-formula move can discover "drop formula 1 entirely, give
    # its share to formula 2" in one step -- pinned_lp can, by pinning
    # formula 0's current rate as a floor and re-solving the LP for
    # everything else (formula 1 included) from scratch.
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0, limit=10.0),
        Formula(consumption=np.array([1.0]), output=1.0, limit=10.0),
        Formula(consumption=np.array([1.0]), output=5.0, limit=10.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([10.0])
    rates = np.array([5.0, 5.0, 0.0])
    rng = __import__("random").Random(1)  # picks k=1, pinned=[0]
    new_rates = _pinned_lp_move(rates, formulas, consumption, supply, rng)
    assert new_rates is not None
    assert new_rates[0] == 5.0  # the pinned floor is always honored
    assert new_rates[2] > 0.0  # the higher-value formula got activated
    usage = consumption @ new_rates
    assert np.all(usage <= supply + 1e-6)


def test_pinned_lp_move_returns_none_when_no_active_formulas():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rates = np.array([0.0])
    rng = __import__("random").Random(0)
    assert _pinned_lp_move(rates, formulas, consumption, supply, rng) is None


def test_pinned_lp_move_rejects_pinning_that_outstrips_its_own_dependency():
    # formula 0 produces resource X (negative coefficient, e.g. a
    # battery); formula 1 consumes it. Pinning formula 1 alone -- without
    # also crediting formula 0's production, since only pinned formulas'
    # rates get folded into the floor -- would need more of X than the
    # floor-only view can supply -- must reject rather than propose an
    # infeasible plan.
    formulas = [
        Formula(consumption=np.array([-1.0]), output=0.0),
        Formula(consumption=np.array([1.0]), output=1.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([0.0])
    rates = np.array([1.0, 1.0])
    rng = __import__("random").Random(4)  # picks k=1, pinned=[1] (the consumer only)
    assert _pinned_lp_move(rates, formulas, consumption, supply, rng) is None


def test_simulated_annealing_can_reallocate_between_competing_consumers():
    # Mirrors test_shift_move_reallocates_a_fully_consumed_resource, but
    # end-to-end through the full SA loop: a fitness function that only
    # rewards running formula 2 (power) can't be satisfied by round_down/
    # round_up/allocate_slack alone, since formula 1 (sell) already
    # claims all of the shared resource at its simplest denominator.
    formulas = [
        Formula(consumption=np.array([-6.0]), output=0.0, limit=2.0),
        Formula(consumption=np.array([6.0]), output=0.0),
        Formula(consumption=np.array([1.5]), output=0.0),
    ]
    supply = np.array([0.0])
    initial = np.array([2.0, 2.0, 0.0])

    def fitness_fn(rates):
        return float(rates[2])

    outcome = simulated_annealing(
        supply, formulas, initial, fitness_fn, SearchConfig(iterations=500, seed=0)
    )
    assert outcome.rates[2] > 0.0


def test_simulated_annealing_never_returns_worse_than_start():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    supply = np.array([10.0])
    initial = np.array([19 / 20])

    def fitness_fn(rates):
        return float(rates[0])  # pure $ maximization: no reason to move

    outcome = simulated_annealing(
        supply, formulas, initial, fitness_fn, SearchConfig(iterations=200, seed=1)
    )
    assert outcome.fitness >= fitness_fn(initial) - 1e-9


def test_simulated_annealing_prefers_simpler_fraction_when_it_pays_off():
    # A single formula; fitness rewards $ output a little, but heavily
    # penalizes non-power-of-2 denominators -- so trading a tiny bit of
    # output for a much simpler fraction should net a higher score.
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    supply = np.array([1.0])
    initial = np.array([19 / 20])  # denominator 20 = 2^2 * 5

    def fitness_fn(rates):
        r = float(rates[0])
        denom = Fraction(r).limit_denominator(1000).denominator
        complexity = 0.0
        remaining = denom
        while remaining % 2 == 0:
            remaining //= 2
        if remaining > 1:
            complexity = 50.0  # any non-power-of-2 factor is very costly
        return r * 10 - complexity

    outcome = simulated_annealing(
        supply, formulas, initial, fitness_fn, SearchConfig(iterations=500, seed=1)
    )
    assert outcome.fitness > fitness_fn(initial)
    found_denom = Fraction(float(outcome.rates[0])).limit_denominator(1000).denominator
    remaining = found_denom
    while remaining % 2 == 0:
        remaining //= 2
    assert remaining == 1  # landed on a pure power-of-2 denominator


def test_integer_rates_valid_accepts_any_whole_number_in_range():
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0, limit=12.0, integer=True)
    ]
    assert _integer_rates_valid(np.array([0.0]), formulas)
    assert _integer_rates_valid(np.array([7.0]), formulas)
    assert _integer_rates_valid(np.array([12.0]), formulas)


def test_integer_rates_valid_rejects_fractional_rate():
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0, limit=12.0, integer=True)
    ]
    assert not _integer_rates_valid(np.array([7.5]), formulas)


def test_integer_rates_valid_ignores_non_integer_formulas():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0, limit=12.0)]
    assert _integer_rates_valid(np.array([7.3]), formulas)


def test_snap_integer_rates_rounds_and_clips():
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0, limit=1.0, integer=True)
    ]
    assert _snap_integer_rates(np.array([0.7]), formulas)[0] == 1.0
    assert _snap_integer_rates(np.array([0.3]), formulas)[0] == 0.0
    assert _snap_integer_rates(np.array([1.9]), formulas)[0] == 1.0  # clipped to limit


def test_toggle_integer_move_turns_a_zero_bonus_fully_on():
    formulas = [
        Formula(consumption=np.array([1.0]), output=2.0, limit=1.0, integer=True)
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rng = __import__("random").Random(0)
    new_rates = _toggle_integer_move(
        np.array([0.0]), formulas, consumption, supply, rng
    )
    assert new_rates is not None
    assert new_rates[0] == 1.0


def test_toggle_integer_move_turns_a_nonzero_bonus_fully_off():
    formulas = [
        Formula(consumption=np.array([1.0]), output=2.0, limit=1.0, integer=True)
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([1.0])
    rng = __import__("random").Random(0)
    new_rates = _toggle_integer_move(
        np.array([1.0]), formulas, consumption, supply, rng
    )
    assert new_rates is not None
    assert new_rates[0] == 0.0


def test_toggle_integer_move_rejects_turning_off_a_producer_a_consumer_depends_on():
    """Mirrors test_round_down_move_rejects_move_that_starves_a_downstream
    _consumer -- turning an integer formula off is a net-producer
    reduction just like round_down, and must be rejected the same way if
    a different, already-fixed formula still depends on what it makes."""
    # formula 0 (integer, toggled) produces 1 unit of resource 0 per
    # multiple; formula 1 (fixed) consumes 1 unit of resource 0.
    formulas = [
        Formula(consumption=np.array([-1.0]), output=0.0, limit=1.0, integer=True),
        Formula(consumption=np.array([1.0]), output=0.0),
    ]
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    supply = np.array([0.0])
    rng = __import__("random").Random(0)
    rates = np.array([1.0, 1.0])  # formula 0 on, supplying formula 1's only input
    # Force the toggle candidate to formula 0 by making it the only
    # integer formula in the list.
    new_rates = _toggle_integer_move(rates, formulas, consumption, supply, rng)
    assert new_rates is None


def test_simulated_annealing_cannot_double_dip_an_integer_bonus_with_extra_flow():
    """Regression for a real bug found via CLI testing: a Formula.integer
    =True, limit=1 "reached the goal in one go" bonus (see
    factorylib.endfield.pp_goals.hard_satisfaction_bonus) was being run
    at a FRACTIONAL rate by the SA search (nothing enforced Formula.
    integer outside the MILP path), silently paying out partial credit
    for a goal that was never actually completed -- and since the
    bonus's pp-per-unit-of-flow ratio was higher than a legitimate tail
    tier's marginal value, this made the search deliberately overshoot a
    hard cap (power production 177% of target) to keep "topping up" the
    exploitable fractional bonus.

    Modeled the way the user framed it: reaching 100% of some flow
    converts (capped at 1 multiple) into one "satisfaction point"; that
    single point can go EITHER to a lump integer bonus (2 reward, all-
    or-nothing) OR to a continuous 1:1 fallback (up to 1 reward), never
    both -- so total achievable reward is capped at 2 regardless of how
    much flow is available beyond the 100% mark, and the bonus itself
    must never be claimed fractionally."""
    target = 100.0
    # resource 0: flow; resource 1: satisfaction_point
    formulas = [
        Formula(consumption=np.array([target, -1.0]), output=0.0, limit=1.0),
        Formula(consumption=np.array([0.0, 1.0]), output=2.0, limit=1.0, integer=True),
        Formula(consumption=np.array([0.0, 1.0]), output=1.0, limit=1.0),
    ]
    outputs = np.array([f.output for f in formulas])

    def fitness_fn(rates):
        return float(rates @ outputs)

    def run(flow_supply):
        supply = np.array([flow_supply, 0.0])
        initial = np.zeros(3)
        return simulated_annealing(
            supply, formulas, initial, fitness_fn, SearchConfig(iterations=2000, seed=0)
        )

    outcome_140 = run(1.4 * target)
    outcome_200 = run(2.0 * target)
    assert outcome_140.fitness == outcome_200.fitness == 2.0
    # the bonus is claimed exactly (1 multiple), never fractionally, and
    # the continuous fallback gets nothing once the bonus wins it all
    assert outcome_140.rates[1] == 1.0
    assert outcome_140.rates[2] == 0.0


def test_search_dispatch_unknown_backend_raises():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    try:
        search(
            np.array([1.0]), formulas, np.array([0.5]), lambda r: 0.0, backend="nope"
        )
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_search_dispatches_to_sa_by_default():
    formulas = [Formula(consumption=np.array([1.0]), output=1.0)]
    supply = np.array([10.0])
    initial = np.array([1.0])
    outcome = search(
        supply,
        formulas,
        initial,
        lambda r: float(r[0]),
        SearchConfig(iterations=10, seed=0),
    )
    assert math.isfinite(outcome.fitness)


def test_scipy_backend_returns_feasible_result():
    formulas = [
        Formula(consumption=np.array([1.0]), output=1.0),
        Formula(consumption=np.array([1.0]), output=1.0),
    ]
    supply = np.array([10.0])
    initial = np.array([5.0, 5.0])
    outcome = scipy_dual_annealing(
        supply,
        formulas,
        initial,
        lambda r: float(np.sum(r)),
        SearchConfig(iterations=50, seed=0),
    )
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    assert np.all(consumption @ outcome.rates <= supply + 1e-6)
