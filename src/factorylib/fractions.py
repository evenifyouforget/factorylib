"""Represent numerical results as simple fractions when possible."""

from __future__ import annotations

import warnings
from fractions import Fraction


def snap_value(
    x: float, max_denom: int = 1000, tol: float = 1e-9, *, warn: bool = True
) -> Fraction:
    """Convert float to nearest simple Fraction.

    Warns if no fraction with denominator <= max_denom is within tol
    (unless warn=False). Useful for snapshotting numerical results (LP
    solutions, breakpoints) as exact fractions for readability and tests.

    Args:
        x: value to convert.
        max_denom: largest denominator to consider.
        tol: maximum allowed difference between x and the returned fraction.
        warn: whether to warn when no close fraction is found.
    """
    f = Fraction(x).limit_denominator(max_denom)
    if warn and abs(float(f) - x) > tol:
        warnings.warn(
            f"snap_value: {x!r} not close to any fraction with denom <= {max_denom} "
            f"(nearest: {f}, diff: {abs(float(f) - x):.2e})"
        )
    return f


def snap_or_float(
    x: float, max_denom: int = 1000, tol: float = 1e-9
) -> Fraction | float:
    """Like snap_value, but silently returns the raw float when no simple
    fraction is within tol, instead of warning.

    Not finding a simple fraction is an expected, valid outcome here (not a
    mistake to flag), so no warning is raised.
    """
    f = Fraction(x).limit_denominator(max_denom)
    if abs(float(f) - x) > tol:
        return x
    return f
