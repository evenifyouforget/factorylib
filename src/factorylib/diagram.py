"""Render a solved :class:`~factorylib.optimize.OptimizeResult` as a Graphviz
production-flow diagram: one node per (non-hidden, non-zero) material, one
node per active recipe that touches 2+ visible materials, and edges showing
each recipe's contribution to/from each material.

Ported (and split out) from ``factorylib.endfield.main``'s monolithic
``optimize()``, which built this graph inline, interleaved with solving and
report printing.
"""

from __future__ import annotations

import textwrap
from typing import Any

import graphviz
from numpy.typing import NDArray

from factorylib.material import HIDDEN, Material
from factorylib.optimize import OptimizeResult, material_balance

LABEL_WIDTH = 40


def wrap_label(text: str) -> str:
    """Word-wrap each line of ``text`` to :data:`LABEL_WIDTH` columns, for
    readable Graphviz node labels."""
    return "\n".join(
        textwrap.fill(line, width=LABEL_WIDTH) for line in text.splitlines()
    )


def build_graph(result: OptimizeResult, multiples: NDArray[Any]) -> graphviz.Digraph:
    """Build (but don't render) the production-flow diagram for a solved
    result at the given (usually fraction-snapped) recipe multiples."""
    dot = graphviz.Digraph(engine="sfdp", graph_attr={"overlap_scaling": "-10"})
    plus_amount, net_amount = material_balance(result, multiples)
    node_names: dict[Material, str] = {}
    hidden_node_mats: set[Material] = set()
    for i, material in enumerate(result.materials):
        node_names[material] = inode_name = f"material{i}"
        iplus = plus_amount[i]
        inet = net_amount[i]
        if (iplus == 0 and inet == 0) or HIDDEN in material.tags:
            hidden_node_mats.add(material)
            continue
        isub = iplus - inet
        dot.node(
            inode_name,
            wrap_label(
                f"{material}\n\n+{iplus}{material.unit} - {isub}{material.unit}"
                f"\n\n={inet}{material.unit}"
            ),
        )
    for i, m in enumerate(multiples):
        if m == 0:
            continue
        recipe = result.recipes[i]
        inode_name = f"recipe{i}"
        nodef = (inode_name, wrap_label(f"{recipe.name}\n\n{m} multiples"))
        edgefs: list[tuple[str, str, str]] = []
        for j, material in enumerate(result.materials):
            per_multiple = result.recipe_matrix[i, j]
            if per_multiple == 0:
                continue
            contribution = m * per_multiple
            if material in hidden_node_mats:
                continue
            edge_in = inode_name
            edge_out = node_names[material]
            if contribution < 0:
                edge_in, edge_out = edge_out, edge_in
                contribution = -contribution
            edgefs.append((edge_in, edge_out, f"{contribution}{material.unit}"))
        if len(edgefs) >= 2:
            # only show recipes that have 2 or more connections
            dot.node(*nodef)
            for edgef in edgefs:
                dot.edge(*edgef)
    return dot


def render_graph(dot: graphviz.Digraph, outfile: str) -> None:
    """Render ``dot`` to ``outfile`` (as in ``graphviz.Digraph.render``)."""
    dot.render(outfile)
