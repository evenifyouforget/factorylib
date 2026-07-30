"""MILP-based recipe-multiple optimization over a Material/Recipe model.

Given a set of :class:`~factorylib.material.Recipe` objects (each an
expression over :class:`~factorylib.material.Material` leaves) and a single
material to maximize, builds a recipe/material incidence matrix and solves
for the non-negative (optionally integer-constrained) run count of each
recipe that maximizes net production of the target material, subject to every
other material's net balance staying non-negative.

This supersedes the previous ``Formula``-based ``maximize_dollar``, a
plain-LP, no-recipe-algebra dollar maximizer with no MILP/integer support.
It's ported and refactored from ``factorylib.endfield.main``'s monolithic
``optimize()``, split into this module's pure "solve" step, then
:mod:`factorylib.report` (text report) and :mod:`factorylib.diagram`
(Graphviz rendering) as separate consumers of the same result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp

from factorylib.material import Material, Recipe, gather_materials, substitute


@dataclass
class OptimizeResult:
    """Result of :func:`solve`.

    Attributes:
        status: raw ``scipy.optimize.milp`` status code (0 == optimal).
        message: solver status message.
        objective: maximized net amount of the target material.
        materials: all materials involved, sorted (index matches
            ``recipe_matrix``'s columns).
        recipes: all recipes considered, in the order supplied (index
            matches ``recipe_matrix``'s rows and ``multiples``).
        recipe_matrix: shape (num_recipes, num_materials); entry [i, j] is
            the net amount of material j produced (negative if consumed)
            per single run of recipe i.
        multiples: shape (num_recipes,); the raw (unsnapped) optimal run
            count for each recipe.
    """

    status: int
    message: str
    objective: float
    materials: list[Material]
    recipes: list[Recipe]
    recipe_matrix: NDArray[np.float64]
    multiples: NDArray[np.float64]


def material_balance(
    result: OptimizeResult, multiples: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Per-material (gross produced, net) amounts for a given set of recipe
    multiples (typically ``result.multiples`` or a snapped variant from
    :func:`factorylib.fractions.snap_multiples`).

    Returns:
        (plus_amount, net_amount): both shape (num_materials,).
    """
    plus_amount = np.maximum(0, result.recipe_matrix.T) @ multiples
    net_amount = result.recipe_matrix.T @ multiples
    return plus_amount, net_amount


def solve(
    all_materials: set[Material] | list[Material],
    all_recipes: list[Recipe],
    material_to_maximize: Material,
) -> OptimizeResult:
    """Solve for the recipe multiples that maximize net production of
    ``material_to_maximize``, subject to every material's net balance
    staying non-negative.

    Raises:
        ValueError: if ``material_to_maximize`` isn't among ``all_materials``
            or any recipe's gathered materials.
    """
    num_recipes = len(all_recipes)
    # Get the complete list of all materials, including recipe counters
    materials = sorted(set(all_materials) | gather_materials(all_recipes))
    num_materials = len(materials)
    # Assign each material a unit basis vector
    subs_dict: dict[Material, NDArray[np.float64]] = {}
    max_objective_index: int | None = None
    for i, material in enumerate(materials):
        a = np.zeros(num_materials, dtype=float)
        a[i] = 1
        subs_dict[material] = a
        if material == material_to_maximize:
            max_objective_index = i
    if max_objective_index is None:
        raise ValueError(
            f"material_to_maximize {material_to_maximize!r} not found among "
            "all_materials or recipes' gathered materials"
        )
    # Construct the recipe matrix
    recipe_matrix = np.zeros((num_recipes, num_materials), dtype=float)
    for i, recipe in enumerate(all_recipes):
        recipe_matrix[i, :] += substitute(recipe.expression, subs_dict)
    # Construct the bounds on the decision variables (recipe multiples)
    lb = np.zeros(num_recipes, dtype=float)
    ub = np.full(num_recipes, np.inf, dtype=float)
    for i, recipe in enumerate(all_recipes):
        ub[i] = recipe.max_multiples
    bounds = Bounds(lb=lb, ub=ub)
    # Construct the integrality flags
    integrality = np.zeros(num_recipes, dtype=int)
    for i, recipe in enumerate(all_recipes):
        if recipe.integer_only:
            integrality[i] = 1
    # Construct the constraints (all net supplies must be non-negative)
    constraints = LinearConstraint(recipe_matrix.T, lb=0)
    # Minimization objective
    c = -recipe_matrix[:, max_objective_index]
    # Query MILP solver. HiGHS's default mip_rel_gap (~1e-4) lets it stop at
    # a solution merely close to optimal rather than proving the true
    # optimum -- fine for small problems, but scenarios with many integer
    # variables (e.g. the "mixed" PP-overlay target's many Award-Points
    # tiers) have enough room in that gap to land noticeably short of the
    # real optimum. Tightening it makes the solve deterministic and
    # actually optimal, at negligible extra cost for problems this size.
    res = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"mip_rel_gap": 1e-9},
    )
    return OptimizeResult(
        status=int(res.status),
        message=str(res.message),
        objective=float(-res.fun),
        materials=materials,
        recipes=list(all_recipes),
        recipe_matrix=recipe_matrix,
        multiples=np.asarray(res.x, dtype=float),
    )
