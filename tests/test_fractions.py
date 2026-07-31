"""Tests for factorylib.fractions
(Fraction2/find_close_fraction/snap_multiples).

This replaces the old test_fractions.py, which tested the now-removed
snap_value/snap_or_float API, superseded by find_close_fraction, ported
from factorylib.endfield.main (the design oracle).
"""

from __future__ import annotations

from fractions import Fraction

from factorylib.fractions import Fraction2, find_close_fraction, snap_multiples


def test_fraction2_str_shows_fraction_and_float():
    f = Fraction2(3, 8)
    assert str(f) == "[3/8 = 0.375]"


def test_find_close_fraction_snaps_clean_value():
    result = find_close_fraction(0.375)
    assert isinstance(result, Fraction2)
    assert result == Fraction(3, 8)


def test_find_close_fraction_returns_original_float_when_not_close():
    # Not force_fractions, and not close to a simple fraction of denom <= 8
    # (after the x12 premultiply): should fall back to the raw float.
    x = 0.3141592653589793
    result = find_close_fraction(x)
    assert result == x
    assert isinstance(result, float)


def test_find_close_fraction_force_true_always_returns_fraction():
    x = 0.3141592653589793
    result = find_close_fraction(x, force_fractions=True)
    assert isinstance(result, Fraction2)


def test_find_close_fraction_force_true_never_exceeds_original_by_default():
    """With allow_greater=False (the default), the snapped fraction must be
    <= the original value (walked down via DECREMENT steps until it fits)."""
    x = 0.4999
    result = find_close_fraction(x, force_fractions=True)
    assert isinstance(result, Fraction2)
    assert float(result) <= x + 1e-9


def test_find_close_fraction_force_true_zero_floor_when_not_allow_negative():
    """Walking a small positive value down without allow_negative bottoms out
    at exactly 0 rather than going negative."""
    result = find_close_fraction(1e-6, force_fractions=True)
    assert result == Fraction2(0)


def test_find_close_fraction_zero_is_exact():
    result = find_close_fraction(0.0)
    assert result == 0


def test_snap_multiples_vectorizes_over_list():
    values = [0.0, 1.0, 0.375, 0.3141592653589793]
    result = snap_multiples(values)
    assert result.dtype == object
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == Fraction(3, 8)
    assert result[3] == values[3]  # unsnapped, falls back to raw float


def test_snap_multiples_force_fractions_all_fractions():
    values = [0.3141592653589793, 0.5, 12.0]
    result = snap_multiples(values, force_fractions=True)
    assert all(isinstance(v, Fraction2) for v in result)
