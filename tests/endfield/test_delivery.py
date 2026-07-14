import numpy as np

from factorylib.endfield.delivery import accumulation_rates, predict_delivery_selections
from factorylib.endfield.wuling import RESOURCE_NAMES


def test_accumulation_rates_includes_resources_with_slack():
    slack = np.zeros(len(RESOURCE_NAMES))
    slack[RESOURCE_NAMES.index("ori")] = 50.0
    rates = accumulation_rates({}, slack)
    assert rates == {"Originium Ore": 50.0}


def test_accumulation_rates_excludes_zero_slack_resources():
    slack = np.zeros(len(RESOURCE_NAMES))
    rates = accumulation_rates({}, slack)
    assert rates == {}


def test_accumulation_rates_includes_sandleaf_powder_scaled_by_good_yield():
    slack = np.zeros(len(RESOURCE_NAMES))
    rates = accumulation_rates({"sandleaf_powder": 2.0}, slack)
    assert rates == {"Sandleaf Powder": 180.0}  # 2 multiples * 90 items/multiple


def test_accumulation_rates_excludes_thermal_bank():
    """thermal_bank produces power (W), not a stashable material."""
    slack = np.zeros(len(RESOURCE_NAMES))
    rates = accumulation_rates({"thermal_bank": 14.0}, slack)
    assert rates == {}


def test_accumulation_rates_includes_unconsumed_components():
    slack = np.zeros(len(RESOURCE_NAMES))
    rates = accumulation_rates({"ferrium_component": 0.5}, slack)
    assert rates == {"Ferrium Component": 3.0}  # 0.5 multiples * 6 items/multiple


def test_predict_delivery_selections_dominant_good_wins():
    # Default DeliverySimConfig runs 100 days; with the depot cap, even a
    # tiny accumulator (1/min = 1440/day) reaches the 80k cap eventually
    # (~day 56) and starts competing -- so use a short window instead of
    # asserting it's never picked at all, which isn't true over the long
    # run once a cap is in play (see delivery.py's module docstring).
    from factorylib.delivery import DeliverySimConfig

    slack = np.zeros(len(RESOURCE_NAMES))
    slack[RESOURCE_NAMES.index("ori")] = 1.0  # tiny, dominated within the window
    tally = predict_delivery_selections(
        {"sandleaf_powder": 10.0}, slack, DeliverySimConfig(simulation_days=10)
    )
    assert tally["Sandleaf Powder"] > 0
    assert tally.get("Originium Ore", 0) == 0


def test_accumulation_rates_excludes_liquids():
    """The depot cannot store liquids at all -- excess Sewage/Xircon
    Effluent/Inert Xircon Effluent causes backpressure, not depot
    accumulation, so they must never be delivery-job candidates."""
    slack = np.zeros(len(RESOURCE_NAMES))
    for liquid in ("sew", "eff", "inert"):
        slack[RESOURCE_NAMES.index(liquid)] = 100.0
    assert accumulation_rates({}, slack) == {}


def test_accumulation_rates_still_includes_solids_alongside_excluded_liquids():
    slack = np.zeros(len(RESOURCE_NAMES))
    slack[RESOURCE_NAMES.index("eff")] = 100.0  # liquid, excluded
    slack[RESOURCE_NAMES.index("dop")] = 55.0  # solid, included
    rates = accumulation_rates({}, slack)
    assert rates == {"Dense Originium Powder": 55.0}
