from collections import deque
from dataclasses import dataclass

import numpy as np

from factorylib.simple import converger_explicit

_EPS = 1e-15


@dataclass
class SolveResult:
    flows: dict
    converged: bool
    reachable: bool

    def __getitem__(self, key):
        return self.flows[key]

    def __contains__(self, key):
        return key in self.flows

    def items(self):
        return self.flows.items()

    def keys(self):
        return self.flows.keys()

    def values(self):
        return self.flows.values()


class Source:
    def __init__(self, supply):
        self.supply = np.asarray(supply, dtype=float)


class SplitterPort:
    def __init__(self, splitter: "Splitter", index: int):
        self.splitter = splitter
        self.index = index


class Splitter:
    def __init__(self, inp, n: int = 2, weights=None, width: float = 1.0):
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            n = len(weights)
            weights = weights / weights.sum()
        else:
            weights = np.full(n, 1.0 / n)
        self.inp = inp
        self.n = n
        self.weights = weights
        self.width = float(width)
        self._ports = [SplitterPort(self, i) for i in range(n)]

    def __getitem__(self, i) -> SplitterPort:
        return self._ports[i]


class Converger:
    def __init__(self, inputs, weights=None, width: float = 1.0):
        self.inputs = list(inputs)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()
        self.weights = weights
        self.width = float(width)

    def set_input(self, index: int, node) -> None:
        self.inputs[index] = node


def _inputs_of(node) -> list:
    if isinstance(node, Source):
        return []
    if isinstance(node, SplitterPort):
        inp = node.splitter.inp
        return [inp] if inp is not None else []
    if isinstance(node, Converger):
        return [inp for inp in node.inputs if inp is not None]
    raise TypeError(f"Unknown node type: {type(node)}")


def _collect_nodes(start) -> list:
    """DFS backward; when a SplitterPort is found, include all sibling ports."""
    visited: set[int] = set()
    result = []
    stack = [start]
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        result.append(node)
        if isinstance(node, SplitterPort):
            for sibling in node.splitter._ports:
                if id(sibling) not in visited:
                    stack.append(sibling)
        for inp in _inputs_of(node):
            stack.append(inp)
    return result


def _topo_sort(nodes: list) -> tuple[list, set[int]]:
    """Kahn's algorithm; returns (order, cycle_node_ids).
    order: sources first, sinks last; cycle nodes appended after in DFS order.
    cycle_node_ids: set of id() for nodes that participate in cycles."""
    by_id = {id(n): n for n in nodes}
    in_degree: dict[int, int] = {id(n): 0 for n in nodes}
    successors: dict[int, list[int]] = {id(n): [] for n in nodes}

    for node in nodes:
        for inp in _inputs_of(node):
            if id(inp) in by_id:
                successors[id(inp)].append(id(node))
                in_degree[id(node)] += 1

    queue = deque(nid for nid in by_id if in_degree[nid] == 0)
    order = []
    while queue:
        nid = queue.popleft()
        order.append(by_id[nid])
        for succ in successors[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    in_order = {id(n) for n in order}
    cycle_node_ids = {id(n) for n in nodes if id(n) not in in_order}

    for n in nodes:
        if id(n) not in in_order:
            order.append(n)

    return order, cycle_node_ids


def _validate_wired(nodes: list) -> None:
    for node in nodes:
        if isinstance(node, SplitterPort):
            if node.splitter.inp is None:
                raise ValueError(
                    "Splitter has unwired input (inp=None)."
                    " Pass inp to constructor or assign it."
                )
        elif isinstance(node, Converger):
            for i, inp in enumerate(node.inputs):
                if inp is None:
                    raise ValueError(
                        f"Converger has unwired input at index {i}."
                        " Call set_input() before solving."
                    )


def solve(node, *, tol: float = 1e-9, max_iter: int = 1000) -> SolveResult:
    """
    Solve steady-state flows for the network rooted at `node`.
    Supports cyclic graphs via iterative fixed-point convergence.

    Returns a SolveResult with:
        flows: dict mapping each node to its actual flow vector (np.ndarray).
            Flow vectors have one dimension per source.
        converged: whether iteration reached tolerance within max_iter.
        reachable: whether steady state is reachable from empty (= converged,
            since iteration always starts from all-zero flows).
    """
    nodes = _collect_nodes(node)
    _validate_wired(nodes)
    topo, cycle_node_ids = _topo_sort(nodes)

    sources = [n for n in nodes if isinstance(n, Source)]
    ndim = sources[0].supply.shape[0] if sources else 1

    seen_splitters: set[int] = set()
    units = []
    for n in topo:
        if isinstance(n, Converger):
            units.append(n)
        elif isinstance(n, SplitterPort):
            s = n.splitter
            if id(s) not in seen_splitters:
                seen_splitters.add(id(s))
                units.append(s)

    flows: dict = {n: np.zeros(ndim) for n in nodes}
    scalar_demand: dict = {n: 1.0 for n in nodes}

    converged = False
    for _ in range(max_iter):
        old_flows = {n: flows[n].copy() for n in nodes}

        # --- Forward pass ---
        splitter_cache: dict[int, np.ndarray] = {}

        for n in topo:
            if isinstance(n, Source):
                total = float(np.sum(n.supply))
                if total > _EPS:
                    flows[n] = n.supply * min(1.0, scalar_demand[n] / total)
                else:
                    flows[n] = n.supply.copy()

            elif isinstance(n, SplitterPort):
                s = n.splitter
                if id(s) not in splitter_cache:
                    in_flow = flows[s.inp]
                    in_avail = float(np.sum(in_flow))
                    in_avail_eff = min(in_avail, s.width)
                    out_caps = np.array([scalar_demand[p] for p in s._ports])
                    if in_avail_eff > _EPS:
                        splitter_cache[id(s)] = (
                            converger_explicit(out_caps / in_avail_eff, s.weights)
                            * in_avail_eff
                        )
                    else:
                        splitter_cache[id(s)] = np.zeros(s.n)
                port_fracs = splitter_cache[id(s)]
                in_flow = flows[s.inp]
                in_avail = float(np.sum(in_flow))
                if in_avail > _EPS:
                    flows[n] = in_flow * (port_fracs[n.index] / in_avail)
                else:
                    flows[n] = np.zeros(ndim)

            elif isinstance(n, Converger):
                in_vecs = [flows[inp] for inp in n.inputs if inp is not None]
                non_none_inputs = [inp for inp in n.inputs if inp is not None]
                in_totals = np.array(
                    [float(np.sum(flows[inp])) for inp in non_none_inputs]
                )
                w = n.weights
                if w is None:
                    k = len(non_none_inputs)
                    w = np.full(k, 1.0 / k) if k > 0 else np.array([])
                cap = min(float(scalar_demand[n]), n.width)
                if cap > _EPS and len(in_vecs) > 0 and np.any(in_totals > _EPS):
                    fracs = converger_explicit(in_totals / cap, w) * cap
                    output = np.zeros(ndim)
                    for frac, in_vec, in_total in zip(fracs, in_vecs, in_totals):
                        if in_total > _EPS:
                            output += (frac / in_total) * in_vec
                    flows[n] = output
                else:
                    flows[n] = np.zeros(ndim)

        # --- Convergence check ---
        max_diff = max(float(np.max(np.abs(flows[n] - old_flows[n]))) for n in nodes)
        if max_diff < tol:
            converged = True
            break

        # --- Backward pass ---
        for unit in reversed(units):
            if isinstance(unit, Converger):
                non_none_inputs = [inp for inp in unit.inputs if inp is not None]
                in_totals = np.array(
                    [float(np.sum(flows[inp])) for inp in non_none_inputs]
                )
                w = unit.weights
                if w is None:
                    k = len(non_none_inputs)
                    w = np.full(k, 1.0 / k) if k > 0 else np.array([])
                cap = min(float(scalar_demand[unit]), unit.width)
                if cap > _EPS and len(non_none_inputs) > 0 and np.any(in_totals > _EPS):
                    fracs = converger_explicit(in_totals / cap, w) * cap
                    frac_iter = iter(fracs)
                    for inp in unit.inputs:
                        if inp is not None:
                            f = float(next(frac_iter))
                            if id(inp) not in cycle_node_ids:
                                scalar_demand[inp] = f
                else:
                    for inp in unit.inputs:
                        if inp is not None and id(inp) not in cycle_node_ids:
                            scalar_demand[inp] = 0.0

            elif isinstance(unit, Splitter):
                in_flow = flows[unit.inp]
                in_avail = float(np.sum(in_flow))
                in_avail_eff = min(in_avail, unit.width)
                out_caps = np.array([scalar_demand[p] for p in unit._ports])
                if in_avail_eff > _EPS:
                    port_fracs = (
                        converger_explicit(out_caps / in_avail_eff, unit.weights)
                        * in_avail_eff
                    )
                else:
                    port_fracs = np.zeros(unit.n)
                if id(unit.inp) not in cycle_node_ids:
                    scalar_demand[unit.inp] = min(float(np.sum(port_fracs)), unit.width)

    return SolveResult(flows=flows, converged=converged, reachable=converged)
