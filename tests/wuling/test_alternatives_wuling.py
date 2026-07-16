"""Regression test: find_alternatives reproduces the documented exact tie
from test_baseline.py::test_wuling_1p2_hetonite_worth at hp = $48."""

import numpy as np

from factorylib.alternatives import find_alternatives

from ._helpers import BASE_INCOME, _make_wuling_formulas, make_formula


def test_hetonite_worth_48_exact_tie():
    f = _make_wuling_formulas()
    f["hp"] = make_formula([0, 0, 30, 240], output=48 * 6)
    f["hx"].limit = 1  # z=7 branch
    income = (
        BASE_INCOME
        + 7 * np.array([30, 0, 0, 0], dtype=float)
        + np.array([0, 50, 0, 0], dtype=float)
    )

    result = find_alternatives(income, list(f.values()))
    assert np.isclose(result.baseline.dollar_output, 2229 / 2)
    assert np.allclose(
        result.baseline.formula_rates, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0]
    )

    hp_on_rates = [53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0]
    assert any(
        np.isclose(alt.dollar_output, 2229 / 2)
        and np.allclose(alt.formula_rates, hp_on_rates)
        for alt in result.alternatives
    )
