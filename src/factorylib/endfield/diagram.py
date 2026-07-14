"""Optional Graphviz diagram of a Wuling production plan: materials as
oval nodes, active (nonzero-rate) formulas as box nodes, edges labeled
with the actual items/min flow between them -- so a plan reads as an
actual factory layout instead of an abstract list of rates and a
separate material-balance listing.

Degrades gracefully if the `graphviz` Python package isn't installed, or
if it's installed but the system `dot` executable isn't on PATH (the
common case: `pip install graphviz` alone only installs the thin Python
wrapper, not the actual Graphviz binaries) -- both cases return None
instead of raising, so a missing optional dependency never breaks the
CLI.
"""

from __future__ import annotations

import os

from factorylib.optimize import Formula

try:
    import graphviz
except ImportError:
    graphviz = None


def _build_graph(
    rates_by_name: dict[str, float],
    formulas: dict[str, Formula],
    resource_names: list[str],
    resource_labels: dict[str, str],
    formula_labels: dict[str, str],
) -> "graphviz.Digraph":
    """Build the Digraph in memory (no rendering) -- kept separate from
    generate_diagram() so the graph structure itself (nodes/edges/labels)
    is testable without a working `dot` executable.

    Only nonzero-rate formulas and the resources they actually touch are
    included, mirroring the CLI's own formula/material-balance listings
    (which already filter to nonzero rates/flows) rather than the full,
    mostly-inactive recipe set.
    """
    active = {name: rate for name, rate in rates_by_name.items() if abs(rate) > 1e-9}

    dot = graphviz.Digraph(comment="Wuling production plan")
    dot.attr(rankdir="LR")

    touched_resources: set[str] = set()
    for name in active:
        formula = formulas.get(name)
        if formula is None:
            continue
        for ri, coeff in enumerate(formula.consumption):
            if abs(coeff) > 1e-9:
                touched_resources.add(resource_names[ri])

    for resource_name in sorted(touched_resources):
        dot.node(
            f"r_{resource_name}",
            resource_labels.get(resource_name, resource_name),
            shape="ellipse",
        )

    for name, rate in active.items():
        formula = formulas.get(name)
        if formula is None:
            continue
        label = formula_labels.get(name, name)
        node_id = f"f_{name}"
        dot.node(node_id, f"{label}\n{rate:.4g}x", shape="box")
        for ri, coeff in enumerate(formula.consumption):
            if abs(coeff) < 1e-9:
                continue
            resource_id = f"r_{resource_names[ri]}"
            flow = rate * coeff
            if coeff > 0:
                dot.edge(resource_id, node_id, label=f"{flow:.4g}/min")
            else:
                dot.edge(node_id, resource_id, label=f"{-flow:.4g}/min")

    return dot


def generate_diagram(
    rates_by_name: dict[str, float],
    formulas: dict[str, Formula],
    resource_names: list[str],
    resource_labels: dict[str, str],
    formula_labels: dict[str, str],
    path: str,
) -> str | None:
    """Render a plan's active formulas and the materials flowing between
    them to `path` (format inferred from its extension, "png" if none is
    given). Returns the path actually written, or None if the `graphviz`
    package or its `dot` executable isn't available.
    """
    if graphviz is None:
        return None
    if not any(abs(rate) > 1e-9 for rate in rates_by_name.values()):
        return None

    dot = _build_graph(
        rates_by_name, formulas, resource_names, resource_labels, formula_labels
    )

    base, ext = os.path.splitext(path)
    dot.format = ext.lstrip(".") or "png"
    try:
        return dot.render(base or path, cleanup=True)
    except graphviz.ExecutableNotFound:
        return None
