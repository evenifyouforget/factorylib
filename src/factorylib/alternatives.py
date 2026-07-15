"""Surface tied/near-tied alternate optimal solutions from maximize_dollar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from factorylib.breakpoints import max_abs_diff
from factorylib.optimize import Formula, OptimizeResult, maximize_dollar


@dataclass
class AlternativesResult:
    """Result of find_alternatives().

    Attributes:
        baseline: the unperturbed optimal solution.
        alternatives: up to max_solutions - 1 structurally distinct tied
            solutions, sorted by dollar_output descending. dollar_output
            and resource_slack are recomputed against the *original*
            (unperturbed) formula outputs, so no epsilon pollutes them.
    """

    baseline: OptimizeResult
    alternatives: list[OptimizeResult]


def find_alternatives(
    supply: np.ndarray | list[float],
    formulas: list[Formula],
    *,
    epsilon: float = 1e-4,
    max_solutions: int = 4,
    rate_tol: float = 1e-6,
    directions: Sequence[np.ndarray] | None = None,
) -> AlternativesResult:
    """Wrap maximize_dollar to find alternate tied optimal vertices.

    Solves the baseline problem, then perturbs formula outputs by +/-
    epsilon along each of a set of perturbation directions (default: the
    signed unit basis, i.e. nudging one formula's output at a time) and
    re-solves. A degenerate LP (multiple optima) will resolve to a
    different vertex under a small nudge; a non-degenerate one won't. Any
    resulting formula_rates that differs from the baseline and from every
    alternative already kept (by more than rate_tol under max_abs_diff)
    AND whose dollar_output, recomputed against the *original*
    (unperturbed) outputs, still matches the baseline's is kept as a
    genuine alternative -- a perturbation direction can otherwise resolve
    to a vertex that's only optimal *for the perturbed problem*, not a
    real tie at all, which the rates-only distinctness check alone can't
    catch.

    Args:
        supply: resource supply vector, as in maximize_dollar.
        formulas: formulas to solve over, as in maximize_dollar.
        epsilon: size of the output perturbation used to break ties.
        max_solutions: maximum total solutions returned (baseline plus
            up to max_solutions - 1 alternatives).
        rate_tol: minimum formula_rates distance for two solutions to be
            considered structurally distinct.
        directions: length-M perturbation vectors to try (each tried as
            both +epsilon*d and -epsilon*d). Defaults to the signed unit
            basis (perturb one formula's output at a time).

    Returns:
        AlternativesResult with the baseline and any distinct alternatives.
    """
    supply_arr = np.asarray(supply, dtype=float)
    baseline = maximize_dollar(supply_arr, formulas)

    if baseline.status != "optimal" or max_solutions <= 1 or not formulas:
        return AlternativesResult(baseline=baseline, alternatives=[])

    consumption = np.stack([f.consumption for f in formulas], axis=1)
    original_outputs = np.array([f.output for f in formulas], dtype=float)

    if directions is None:
        directions = list(np.eye(len(formulas)))

    found: list[OptimizeResult] = []
    for d in directions:
        d = np.asarray(d, dtype=float)
        for sign in (1.0, -1.0):
            delta = sign * epsilon * d
            perturbed = []
            changed = False
            for f, dv in zip(formulas, delta):
                new_output = max(0.0, f.output + dv)
                if new_output != f.output:
                    changed = True
                perturbed.append(Formula(f.consumption, new_output, f.limit))
            if not changed:
                continue

            res = maximize_dollar(supply_arr, perturbed)
            if res.status != "optimal":
                continue
            if max_abs_diff(res.formula_rates, baseline.formula_rates) <= rate_tol:
                continue
            if any(
                max_abs_diff(res.formula_rates, kept.formula_rates) <= rate_tol
                for kept in found
            ):
                continue

            rates = res.formula_rates
            dollar = float(rates @ original_outputs)
            if not np.isclose(dollar, baseline.dollar_output, rtol=1e-6, atol=1e-6):
                continue
            slack = np.maximum(0.0, supply_arr - consumption @ rates)
            found.append(
                OptimizeResult(
                    status="optimal",
                    dollar_output=dollar,
                    formula_rates=rates,
                    resource_slack=slack,
                )
            )

    found.sort(key=lambda r: r.dollar_output, reverse=True)
    return AlternativesResult(
        baseline=baseline, alternatives=found[: max_solutions - 1]
    )
