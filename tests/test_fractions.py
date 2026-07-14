import math
import warnings
from fractions import Fraction

import pytest

from factorylib.fractions import snap_or_float, snap_value


def test_snap_value_clean_fraction():
    assert snap_value(2.5) == Fraction(5, 2)


def test_snap_value_warns_on_no_close_fraction():
    with pytest.warns(UserWarning, match="snap_value"):
        snap_value(math.pi, max_denom=10)


def test_snap_value_warn_false_suppresses_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        snap_value(math.pi, max_denom=10, warn=False)


def test_snap_or_float_returns_fraction_for_clean_value():
    assert snap_or_float(2.5) == Fraction(5, 2)


def test_snap_or_float_returns_raw_float_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = snap_or_float(math.pi, max_denom=10)
    assert result == math.pi
    assert isinstance(result, float)
