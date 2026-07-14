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
    slack = np.zeros(len(RESOURCE_NAMES))
    slack[RESOURCE_NAMES.index("ori")] = 1.0  # tiny, should never win
    tally = predict_delivery_selections({"sandleaf_powder": 10.0}, slack)
    assert tally["Sandleaf Powder"] > 0
    assert tally.get("Originium Ore", 0) == 0
