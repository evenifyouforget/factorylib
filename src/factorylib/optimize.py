"""LP-based dollar maximization for factory resource allocation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

_STATUS_MAP = {
    0: "optimal",
    1: "limit",
    2: "infeasible",
    3: "unbounded",
}


@dataclass
class Formula:
    """
    One production formula consuming resources and producing $ output.

    Args:
        consumption: shape (N,) float array, resource consumption per unit run.
        output: $ produced per unit run (must be >= 0).
        limit: maximum run rate; np.inf means unbounded.
        integer: if True, this formula's rate is constrained to whole
            multiples only (e.g. "how many Forge of the Sky units go to
            Xiranite supply" is inherently a discrete choice, not a
            continuous one). Mixing a handful of integer formulas among
            otherwise-continuous ones turns maximize_dollar's LP into a
            small MILP (see its docstring) -- solved once, exactly,
            instead of the caller having to brute-force enumerate the
            discrete choices itself.
    """

    consumption: np.ndarray
    output: float
    limit: float = field(default=np.inf)
    integer: bool = False

    def __post_init__(self) -> None:
        self.consumption = np.asarray(self.consumption, dtype=float)
        if self.consumption.ndim != 1:
            raise ValueError("consumption must be a 1-D array")
        if self.output < 0:
            raise ValueError("output must be non-negative")
        if self.limit < 0:
            raise ValueError("limit must be non-negative (use np.inf for unbounded)")


@dataclass
class OptimizeResult:
    """
    Result of maximize_dollar().

    Attributes:
        status: "optimal", "infeasible", "unbounded", "zero", or "limit".
        dollar_output: Maximum $ per second achieved.
        formula_rates: shape (M,) optimal run rate for each formula.
        resource_slack: shape (N,) unused supply for each resource (clipped >= 0).
    """

    status: str
    dollar_output: float
    formula_rates: np.ndarray
    resource_slack: np.ndarray


def maximize_dollar(
    supply: np.ndarray | list[float],
    formulas: list[Formula],
) -> OptimizeResult:
    """
    Maximize total $ output subject to resource supply constraints.

    Solves the LP:
        maximize   sum_j output_j * c_j
        subject to sum_j consumption[i,j] * c_j <= supply[i]  for all i
                   0 <= c_j <= limit_j                         for all j

    If any formula has integer=True, this becomes a small MILP (solved via
    scipy.optimize.milp, branch-and-bound) instead of a plain continuous
    LP -- e.g. a discrete allocation choice (how many Forge of the Sky
    units go to Xiranite supply vs. Heavy Xiranite capacity; which
    Metatransfer option to pick) can be modeled as ordinary Formula
    entries (via a couple of extra virtual resource dimensions -- see
    factorylib.endfield.wuling's module docstring) instead of the caller
    having to brute-force enumerate the discrete choices in an outer
    loop. A nice side effect: factorylib.alternatives.find_alternatives'
    epsilon-perturbation tie-finder then also discovers ties *between*
    discrete choices for free, the same way it already finds ties between
    continuous formulas -- both are just Formula objects to it.

    Args:
        supply:   shape (N,) non-negative resource supply vector.
        formulas: list of M Formula objects, all with consumption of length N.

    Returns:
        OptimizeResult with status, dollar_output, formula_rates, resource_slack.

    Raises:
        ValueError: if supply is not 1-D, if supply contains negative values, or
                    if any formula has consumption length != N.
    """
    supply = np.asarray(supply, dtype=float)
    if supply.ndim != 1:
        raise ValueError("supply must be a 1-D array")
    if np.any(supply < 0):
        raise ValueError("supply values must be non-negative")

    N = supply.shape[0]
    M = len(formulas)

    for j, f in enumerate(formulas):
        if f.consumption.shape[0] != N:
            raise ValueError(
                f"Formula {j} has consumption length {f.consumption.shape[0]}, "
                f"expected {N}"
            )

    # Fast path: no formulas or (resources exist and all supply is zero)
    if M == 0 or (N > 0 and np.all(supply == 0)):
        return OptimizeResult(
            status="zero",
            dollar_output=0.0,
            formula_rates=np.zeros(M),
            resource_slack=supply.copy(),
        )

    # Build LP matrices
    # consumption shape: (N, M) — row = resource, col = formula
    consumption = np.stack([f.consumption for f in formulas], axis=1)
    c_obj = -np.array([f.output for f in formulas], dtype=float)
    bounds = [(0.0, None if np.isinf(f.limit) else float(f.limit)) for f in formulas]

    if any(f.integer for f in formulas):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        integrality = np.array([1 if f.integer else 0 for f in formulas])
        result = milp(
            c_obj,
            constraints=LinearConstraint(consumption, -np.inf, supply),
            bounds=Bounds(lb, ub),
            integrality=integrality,
        )
    else:
        result = linprog(
            c_obj,
            A_ub=consumption,
            b_ub=supply,
            bounds=bounds,
            method="highs",
        )

    status_str = _STATUS_MAP.get(result.status, f"solver_status_{result.status}")

    if result.status == 0:
        rates = result.x
        dollar = float(-result.fun)
        slack = np.maximum(0.0, supply - consumption @ rates)
    else:
        rates = np.zeros(M)
        dollar = 0.0
        slack = supply.copy()

    return OptimizeResult(
        status=status_str,
        dollar_output=dollar,
        formula_rates=rates,
        resource_slack=slack,
    )
