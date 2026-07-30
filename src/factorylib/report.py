"""Render an :class:`~factorylib.optimize.OptimizeResult` as a
markdown-ish text report: solver status, which recipes ran at what
multiple, and a per-material balance sheet of where each material's
supply came from.

Ported (and split out) from ``factorylib.endfield.main``'s monolithic
``optimize()``, which built this same report inline via ``print()``
calls interleaved with graph construction.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from factorylib.optimize import OptimizeResult, material_balance


def format_report(result: OptimizeResult, multiples: NDArray[Any]) -> str:
    """Build the full text report for a solved result at the given (usually
    fraction-snapped) recipe multiples."""
    lines = [
        f"# Result {result.status}: {result.message}",
        f"- Maximized score: {result.objective}",
        "## Recipes Used",
    ]
    for i, m in enumerate(multiples):
        if m == 0:
            continue
        lines.append(f"- {m} multiples of {result.recipes[i].name}")
    lines.append("## Balance Sheet Per Material")
    _plus_amount, net_amount = material_balance(result, multiples)
    bits: list[list[str]] = [
        [f"### {material.name} (net {net}{material.unit})"]
        for material, net in zip(result.materials, net_amount)
    ]
    for i, m in enumerate(multiples):
        if m == 0:
            continue
        recipe = result.recipes[i]
        for j, material in enumerate(result.materials):
            per_multiple = result.recipe_matrix[i, j]
            if per_multiple == 0:
                continue
            contribution = m * per_multiple
            bits[j].append(
                f"- {contribution}{material.unit} from {m} multiples of {recipe.name}"
            )
    bits.sort()
    lines.append("\n".join("\n".join(bit) for bit in bits))
    return "\n".join(lines)


def print_report(result: OptimizeResult, multiples: NDArray[Any]) -> None:
    """Print :func:`format_report`'s output, matching the original inline
    ``print()`` sequence from ``factorylib.endfield.main``'s ``optimize()``."""
    print(format_report(result, multiples))
