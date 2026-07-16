"""Optional Graphviz diagram of a Wuling production plan: materials as
oval nodes, active (nonzero-rate) formulas as box nodes, edges labeled
with the actual items/min flow between them -- so a plan reads as an
actual factory layout instead of an abstract list of rates and a
separate material-balance listing.

Degrades gracefully if the `graphviz` Python package isn't installed at
all (returns None) -- but if it's installed and only the system `dot`
executable is missing from PATH (the common case: `pip install
graphviz`/the "diagram" extra only installs the thin Python wrapper, not
the actual Graphviz binaries, which aren't distributed on PyPI at all
and must come from the OS package manager -- see README.md), the raw
`.dot` source is written instead of the rendered image: still a usable
artifact (paste it into any online Graphviz viewer, or render it later
once `dot` is installed), rather than nothing at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from factorylib.optimize import Formula

try:
    import graphviz
except ImportError:
    graphviz = None


@dataclass
class DiagramResult:
    """Result of generate_diagram().

    Attributes:
        path: the file actually written.
        rendered: True if `path` is the rendered image itself; False if
            it's the raw `.dot` source instead (the system `dot`
            executable wasn't available to render it -- see module
            docstring).
    """

    path: str
    rendered: bool


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
) -> DiagramResult | None:
    """Render a plan's active formulas and the materials flowing between
    them to `path` (format inferred from its extension, "png" if none is
    given).

    Returns None only if the `graphviz` Python package itself isn't
    installed, or the plan has no active formulas at all. If the
    package is present but the system `dot` executable isn't (see
    module docstring), falls back to writing the raw `.dot` source next
    to `path` instead (DiagramResult.rendered=False) rather than
    producing nothing.
    """
    if graphviz is None:
        return None
    if not any(abs(rate) > 1e-9 for rate in rates_by_name.values()):
        return None

    dot = _build_graph(
        rates_by_name, formulas, resource_names, resource_labels, formula_labels
    )

    base, ext = os.path.splitext(path)
    dirname = os.path.dirname(base or path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    dot.format = ext.lstrip(".") or "png"
    try:
        rendered_path = dot.render(base or path, cleanup=True)
        return DiagramResult(path=rendered_path, rendered=True)
    except graphviz.ExecutableNotFound:
        # render() already wrote its source to `base or path` (the
        # default source filename) before discovering `dot` is missing
        # -- cleanup=True only deletes that after a *successful* render,
        # so it's left behind here. Remove that half-finished duplicate
        # before writing our own explicit .dot fallback via save().
        leftover_source = base or path
        if os.path.exists(leftover_source):
            os.remove(leftover_source)
        source_path = dot.save(f"{base or path}.dot")
        return DiagramResult(path=source_path, rendered=False)
