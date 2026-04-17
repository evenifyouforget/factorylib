import numpy as np
import pytest

from factorylib.optimize import Formula, OptimizeResult, maximize_dollar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_formula(consumption, output, limit=np.inf):
    return Formula(consumption=np.array(consumption, dtype=float), output=output, limit=limit)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_single_formula_supply_binding():
    """Supply constraint is tighter than the unbounded formula limit."""
    f = make_formula([2.0], output=3.0)
    res = maximize_dollar([5.0], [f])
    assert res.status == "optimal"
    assert np.isclose(res.formula_rates[0], 2.5)
    assert np.isclose(res.dollar_output, 7.5)


def test_single_formula_limit_binding():
    """Formula limit is tighter than the abundant supply."""
    f = make_formula([1.0], output=4.0, limit=1.0)
    res = maximize_dollar([100.0], [f])
    assert res.status == "optimal"
    assert np.isclose(res.formula_rates[0], 1.0)
    assert np.isclose(res.dollar_output, 4.0)


def test_competing_formulas_all_to_better():
    """Two formulas compete for one resource; LP should prefer higher $/unit."""
    # Formula 0: 1 unit resource → $1  (worse)
    # Formula 1: 1 unit resource → $2  (better)
    f0 = make_formula([1.0], output=1.0)
    f1 = make_formula([1.0], output=2.0)
    res = maximize_dollar([6.0], [f0, f1])
    assert res.status == "optimal"
    assert np.isclose(res.formula_rates[0], 0.0, atol=1e-9)
    assert np.isclose(res.formula_rates[1], 6.0)
    assert np.isclose(res.dollar_output, 12.0)


def test_non_competing_formulas():
    """Each formula uses a different resource; both run at full supply."""
    f0 = make_formula([1.0, 0.0], output=1.0)
    f1 = make_formula([0.0, 1.0], output=2.0)
    res = maximize_dollar([10.0, 8.0], [f0, f1])
    assert res.status == "optimal"
    assert np.allclose(res.formula_rates, [10.0, 8.0])
    assert np.isclose(res.dollar_output, 26.0)


def test_two_resources_both_binding():
    """
    2 resources [4, 6], 2 formulas.
    Manual solution: 2c0+c1=4, c0+3c1=6  →  c0=6/5, c1=8/5, dollar=10.
    """
    f0 = make_formula([2.0, 1.0], output=3.0)
    f1 = make_formula([1.0, 3.0], output=4.0)
    res = maximize_dollar([4.0, 6.0], [f0, f1])
    assert res.status == "optimal"
    assert np.allclose(res.formula_rates, [6 / 5, 8 / 5], atol=1e-9)
    assert np.isclose(res.dollar_output, 10.0)


def test_inf_limit_supply_sole_constraint():
    """np.inf limit → None bound in scipy; supply is the only binding constraint."""
    f = make_formula([1.0, 2.0], output=1.0, limit=np.inf)
    res = maximize_dollar([3.0, 5.0], [f])
    assert res.status == "optimal"
    # min(3/1, 5/2) = 2.5
    assert np.isclose(res.formula_rates[0], 2.5)
    assert np.isclose(res.dollar_output, 2.5)


def test_resource_slack_correct():
    """Unused supply is returned in resource_slack."""
    f = make_formula([1.0, 0.0], output=1.0)
    res = maximize_dollar([4.0, 10.0], [f])
    assert res.status == "optimal"
    assert np.isclose(res.resource_slack[0], 0.0, atol=1e-9)
    assert np.isclose(res.resource_slack[1], 10.0)


# ---------------------------------------------------------------------------
# Fast path / edge cases
# ---------------------------------------------------------------------------

def test_zero_supply_fast_path():
    f = make_formula([1.0], output=5.0)
    res = maximize_dollar([0.0], [f])
    assert res.status == "zero"
    assert res.dollar_output == 0.0
    assert np.allclose(res.formula_rates, [0.0])


def test_empty_formulas_fast_path():
    res = maximize_dollar([5.0, 3.0], [])
    assert res.status == "zero"
    assert res.dollar_output == 0.0
    assert res.formula_rates.shape == (0,)


def test_formula_output_zero():
    """Zero-output formula should yield zero revenue, status still optimal."""
    f = make_formula([1.0], output=0.0)
    res = maximize_dollar([10.0], [f])
    assert res.status == "optimal"
    assert res.dollar_output == 0.0


# ---------------------------------------------------------------------------
# Parametrized linearity check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0])
def test_supply_scaling(scale):
    """dollar_output should scale linearly with supply."""
    f = make_formula([1.0], output=3.0)
    res = maximize_dollar([scale], [f])
    assert res.status == "optimal"
    assert np.isclose(res.dollar_output, 3.0 * scale)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_negative_supply_raises():
    f = make_formula([1.0], output=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        maximize_dollar([-1.0, 2.0], [f, f])


def test_2d_supply_raises():
    f = make_formula([1.0], output=1.0)
    with pytest.raises(ValueError, match="1-D"):
        maximize_dollar([[1.0, 2.0]], [f])


def test_consumption_length_mismatch_raises():
    f = make_formula([1.0, 2.0], output=1.0)  # length 2
    with pytest.raises(ValueError, match="consumption length"):
        maximize_dollar([1.0], [f])  # supply length 1


def test_formula_negative_output_raises():
    with pytest.raises(ValueError):
        make_formula([1.0], output=-1.0)


def test_formula_zero_limit_raises():
    with pytest.raises(ValueError):
        make_formula([1.0], output=1.0, limit=0.0)


def test_formula_negative_limit_raises():
    with pytest.raises(ValueError):
        make_formula([1.0], output=1.0, limit=-5.0)
