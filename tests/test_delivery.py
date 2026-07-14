from factorylib.delivery import DeliverySimConfig, simulate_delivery_selections


def test_dominant_material_gets_selected_every_time():
    rates = {"cheap": 100.0, "rare": 0.001}
    tally = simulate_delivery_selections(rates, DeliverySimConfig(simulation_days=10))
    assert tally["cheap"] == 20  # 2/day default * 10 days
    assert tally["rare"] == 0


def test_non_accumulating_materials_are_dropped():
    rates = {"positive": 50.0, "zero": 0.0, "negative": -5.0}
    tally = simulate_delivery_selections(rates)
    assert set(tally) == {"positive"}


def test_empty_input_returns_empty_tally():
    assert simulate_delivery_selections({}) == {}


def test_startup_days_lets_immediate_jobs_find_something_to_select():
    # With zero startup, the depot is empty for day 1's jobs (nothing to
    # select yet -- daily production is only added *after* that day's
    # jobs run). A positive startup period pre-fills the depot so the
    # very first jobs can already succeed.
    rates = {"only": 10.0}
    no_startup = simulate_delivery_selections(
        rates,
        DeliverySimConfig(
            simulation_days=1, startup_days=0.0, jobs_per_day=2, box_capacity=1.0
        ),
    )
    with_startup = simulate_delivery_selections(
        rates,
        DeliverySimConfig(
            simulation_days=1, startup_days=5.0, jobs_per_day=2, box_capacity=1.0
        ),
    )
    assert sum(with_startup.values()) > sum(no_startup.values())


def test_two_competing_materials_split_by_relative_rate():
    # "fast" accumulates twice as quickly as "slow" -> should be picked
    # roughly twice as often over a long enough simulation.
    rates = {"fast": 20.0, "slow": 10.0}
    tally = simulate_delivery_selections(rates, DeliverySimConfig(simulation_days=2000))
    assert tally["fast"] > tally["slow"]


def test_box_capacity_affects_selection_frequency():
    # rate=100/min -> 144000/day. A box capacity small relative to that
    # never depletes the depot, so both daily jobs always succeed. A box
    # capacity comparable to (or bigger than) the daily gain can push the
    # depot negative, causing some jobs to find nothing to select.
    rates = {"only": 100.0}
    small_box = simulate_delivery_selections(
        rates, DeliverySimConfig(simulation_days=50, box_capacity=1000.0)
    )
    large_box = simulate_delivery_selections(
        rates, DeliverySimConfig(simulation_days=50, box_capacity=200_000.0)
    )
    assert small_box["only"] > large_box["only"]


def test_depot_capacity_caps_growth():
    # Both materials vastly exceed the depot cap's replenishment need, so
    # both sit AT the cap essentially the whole simulation -- a genuine,
    # persistent tie, not just occasional overlap.
    rates = {"dominant": 1000.0, "also_capped": 500.0}
    tally = simulate_delivery_selections(
        rates,
        DeliverySimConfig(simulation_days=200, depot_capacity=80_000.0, seed=0),
    )
    assert tally["also_capped"] > 0  # would be 0 forever without a cap


def test_tied_materials_split_roughly_evenly_over_many_trials():
    # Two materials with the same accumulation rate should get picked
    # about equally often once both reach the depot cap.
    rates = {"a": 1000.0, "b": 1000.0}
    tally = simulate_delivery_selections(
        rates,
        DeliverySimConfig(simulation_days=1000, depot_capacity=80_000.0, seed=1),
    )
    total = tally["a"] + tally["b"]
    assert 0.4 * total < tally["a"] < 0.6 * total


def test_tie_break_is_reproducible_with_same_seed():
    rates = {"a": 1000.0, "b": 1000.0}
    first = simulate_delivery_selections(
        rates, DeliverySimConfig(simulation_days=100, seed=42)
    )
    second = simulate_delivery_selections(
        rates, DeliverySimConfig(simulation_days=100, seed=42)
    )
    assert first == second


def test_different_seeds_can_break_ties_differently():
    # With jobs_per_day=2 and two symmetric materials, the totals are
    # forced to an exact 50/50 split every day regardless of the random
    # tie-break (whichever loses job 1's coin flip is deterministically
    # picked by job 2, since it's then the sole max) -- a real property
    # of the simulation, not a bug, but it means the *aggregate* tally
    # can't be used to detect seed sensitivity. jobs_per_day=1 avoids
    # that self-cancelling pairing and exposes real seed-to-seed variance.
    rates = {"a": 1000.0, "b": 1000.0}
    tallies = [
        simulate_delivery_selections(
            rates, DeliverySimConfig(simulation_days=50, jobs_per_day=1, seed=s)
        )
        for s in range(10)
    ]
    assert len({t["a"] for t in tallies}) > 1
