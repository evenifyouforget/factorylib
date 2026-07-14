import math
from fractions import Fraction

import numpy as np

from factorylib.optimize import Formula
from factorylib.search import (
    SearchConfig,
    _allocate_slack_move,
    _round_down_move,
    _round_up_move,
    headroom_loss,
    scipy_dual_annealing,
    search,
    simulated_annealing,
)


def test_round_down_move_reduces_rate_and_simplifies_denominator():
    rng = __import__("random").Random(0)
    rates = np.array([19 / 96, 0.0])
    new_rates = _round_down_move(rates, (1, 2, 3, 4, 6, 8, 12, 24, 48), rng)
    assert new_rates is not None
    assert new_rates[0] <= rates[0]
    new_denom = Fraction(new_rates[0]).limit_denominator(1000).denominator
    assert new_denom < 96


def test_round_down_move_returns_none_when_all_rates_zero():
    rng = __import__("random").Random(0)
    rates = np.array([0.0, 0.0])
    assert _round_down_move(rates, (1, 2, 3), rng) is None


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
