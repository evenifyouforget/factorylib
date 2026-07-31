"""Snap near-fraction floating point results to readable exact fractions.

LP/MILP solutions are floats, but many real solutions land near a "nice"
fraction (e.g. an eighth of a building's throughput). This module finds
such fractions for display purposes, optionally forcing every value to
some close fraction -- trading a small approximation error for a much
more buildable/readable plan.

Ported from ``factorylib.endfield.main``'s
``Fraction2``/``find_close_fraction`` (the design oracle). The loop
structure and comparisons below are intentionally left exactly as
authored there, including comparing against the original ``x`` (not the
running ``x_modified``) on each iteration -- that is what makes the walk
converge monotonically from the un-modified value's nearest fraction
downward.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-12
FRACTION_PREMULTIPLY = 12
FRACTION_LIMIT_DENOM = 8
DECREMENT = 1 / FRACTION_PREMULTIPLY / FRACTION_LIMIT_DENOM**2


class Fraction2(Fraction):
    """Fraction with customized printing: shows both the fraction and its
    float value, e.g. ``[3/8 = 0.375]``.
    """

    def __str__(self) -> str:
        as_fraction = Fraction.__str__(self)
        as_float = float(self)
        return f"[{as_fraction} = {as_float}]"


def find_close_fraction(
    x: float,
    force_fractions: bool = False,
    allow_greater: bool = False,
    allow_negative: bool = False,
) -> Fraction2 | float:
    """Find a close, readable fraction for ``x``.

    force_fractions=False mode: try to find a close fraction, or else
    return the original value unchanged.
    force_fractions=True mode: always returns a fraction, subject to the
    allow_greater/allow_negative constraints (walking the value down in
    small decrements until a fraction that satisfies them is found).
    """

    def round_to_fraction(v: float) -> Fraction2:
        return Fraction2(
            Fraction2(v * FRACTION_PREMULTIPLY).limit_denominator(FRACTION_LIMIT_DENOM)
            / FRACTION_PREMULTIPLY
        )

    x_as_frac = round_to_fraction(x)
    if not force_fractions:
        if np.isclose(float(x_as_frac), x, rtol=_EPS, atol=_EPS):
            return x_as_frac
        return x
    x_modified = x
    while not allow_greater and x_as_frac >= x + _EPS:
        x_modified -= DECREMENT
        if not allow_negative and x_modified < 0:
            return Fraction2(0)
        x_as_frac = round_to_fraction(x_modified)
    return x_as_frac


def snap_multiples(
    values: Iterable[float], force_fractions: bool = False
) -> NDArray[Any]:
    """Vectorized :func:`find_close_fraction` over an array of recipe
    multiples, as produced by :func:`factorylib.optimize.solve`.
    """
    return np.array(
        [find_close_fraction(x, force_fractions=force_fractions) for x in values],
        dtype=object,
    )
