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
