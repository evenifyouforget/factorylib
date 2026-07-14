"""Find discontinuities (breakpoints) of a single-input function.

A breakpoint is a location where a discrete decision flips (e.g. one
production formula turning off in favor of another) as some parameter is
varied. The objective value itself is often continuous through such a
point, but a derived quantity of interest (a specific rate, an active-set
indicator, an outer discrete search's chosen branch) genuinely jumps. This
module locates such jumps for arbitrary single-input functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from factorylib.fractions import snap_or_float


def max_abs_diff(a: Any, b: Any) -> float:
    """Default distance metric: elementwise Chebyshev (max-abs) difference.

    Accepts scalars, tuples/lists, or np.ndarray of matching shape.
    """
    return float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


@dataclass
class Breakpoint:
    """A located discontinuity of g on some bounded interval."""

    x: float
    x_snapped: Fraction | float
    jump: float
    value_below: Any
    value_above: Any


def _refine(
    g: Callable[[float], Any],
    a: float,
    b: float,
    ga: Any,
    gb: Any,
    *,
    min_jump: float,
    x_tol: float,
    max_bisect_iters: int,
    distance: Callable[[Any, Any], float],
) -> tuple[float, float, Any, Any] | None:
    """Bisect [a, b] to narrow down a candidate jump.

    At each step, keep whichever half retains the larger distance between
    its endpoints (that's where the jump lives). A genuine discontinuity's
    jump magnitude does not shrink as the bracket narrows; a merely-steep
    continuous region's does, so it gets discarded as a false positive.
    """
    for _ in range(max_bisect_iters):
        if (b - a) < x_tol:
            break
        mid = (a + b) / 2
        gm = g(mid)
        d_left = distance(ga, gm)
        d_right = distance(gm, gb)
        if d_left >= d_right:
            b, gb = mid, gm
        else:
            a, ga = mid, gm

    if distance(ga, gb) < min_jump:
        return None
    return a, b, ga, gb


def find_breakpoints(
    g: Callable[[float], Any],
    lo: float,
    hi: float,
    *,
    coarse_steps: int = 200,
    min_jump: float = 1e-6,
    x_tol: float = 1e-9,
    max_bisect_iters: int = 60,
    max_denom: int = 1000,
    snap_tol: float = 1e-9,
    distance: Callable[[Any, Any], float] = max_abs_diff,
) -> list[Breakpoint]:
    """Find all jump discontinuities of g on [lo, hi] to numerical precision.

    g may return a float, int, or any array-like (e.g. a specific formula's
    rate, an outer search's chosen discrete branch, or a whole rates
    vector). `distance` measures how different two g-outputs are.

    Algorithm: coarse-scan [lo, hi] into `coarse_steps` cells; for every
    adjacent pair whose distance exceeds `min_jump`, bisect the cell to
    locate the jump precisely (see `_refine`), discarding false positives
    where the apparent jump was just a steep continuous region. Each
    surviving location is snapped to the nearest simple fraction (see
    `factorylib.fractions.snap_or_float`) when the fit is close, else left
    as a raw float.

    Returns breakpoints sorted by x ascending. One call locates every
    breakpoint in the range (not just one) as long as `coarse_steps` is
    fine enough to isolate them into separate cells.
    """
    xs = np.linspace(lo, hi, coarse_steps + 1)
    ys = [g(float(x)) for x in xs]

    out: list[Breakpoint] = []
    for i in range(coarse_steps):
        a, b, ga, gb = float(xs[i]), float(xs[i + 1]), ys[i], ys[i + 1]
        if distance(ga, gb) < min_jump:
            continue
        refined = _refine(
            g,
            a,
            b,
            ga,
            gb,
            min_jump=min_jump,
            x_tol=x_tol,
            max_bisect_iters=max_bisect_iters,
            distance=distance,
        )
        if refined is None:
            continue
        a, b, ga, gb = refined
        x_star = (a + b) / 2
        out.append(
            Breakpoint(
                x=x_star,
                x_snapped=snap_or_float(x_star, max_denom, snap_tol),
                jump=distance(ga, gb),
                value_below=ga,
                value_above=gb,
            )
        )

    return sorted(out, key=lambda bp: bp.x)
