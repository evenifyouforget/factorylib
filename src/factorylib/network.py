from collections import deque

import numpy as np

from factorylib.simple import converger_explicit

_EPS = 1e-15


class Source:
    def __init__(self, supply):
        self.supply = np.asarray(supply, dtype=float)


class SplitterPort:
    def __init__(self, splitter: "Splitter", index: int):
        self.splitter = splitter
        self.index = index


class Splitter:
    def __init__(self, inp, n: int = 2):
        self.inp = inp
        self.n = n
        self._ports = [SplitterPort(self, i) for i in range(n)]

    def __getitem__(self, i) -> SplitterPort:
        return self._ports[i]


class Converger:
    def __init__(self, inputs):
        self.inputs = list(inputs)


def _inputs_of(node) -> list:
    if isinstance(node, Source):
        return []
    if isinstance(node, SplitterPort):
        return [node.splitter.inp]
    if isinstance(node, Converger):
        return node.inputs
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


def _topo_sort(nodes: list) -> list:
    """Kahn's algorithm; returns sources first, sinks last."""
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

    return order


def solve(node, *, tol: float = 1e-9, max_iter: int = 100) -> dict:
    """
    Solve steady-state flows for the DAG rooted at `node`.

    Returns a dict mapping each node to its actual flow vector (np.ndarray).
    Flow vectors have one dimension per source; each source injects a basis
    vector so the composition is readable directly from the result.
    """
    nodes = _collect_nodes(node)
    topo = _topo_sort(nodes)

    sources = [n for n in nodes if isinstance(n, Source)]
    ndim = sources[0].supply.shape[0] if sources else 1

    # Collect unique splitters in topological order (for backward pass)
    seen_splitters: set[int] = set()
    units = []  # Splitter or Converger, in topo order
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

    for _ in range(max_iter):
        old_flows = {n: flows[n].copy() for n in nodes}

        # --- Forward pass ---
        splitter_cache: dict[int, np.ndarray] = {}  # id(splitter) → port_fracs

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
                    out_caps = np.array([scalar_demand[p] for p in s._ports])
                    if in_avail > _EPS:
                        splitter_cache[id(s)] = (
                            converger_explicit(out_caps / in_avail) * in_avail
                        )
                    else:
                        splitter_cache[id(s)] = np.zeros(s.n)
                port_fracs = splitter_cache[id(s)]
                in_flow = flows[n.splitter.inp]
                in_avail = float(np.sum(in_flow))
                if in_avail > _EPS:
                    flows[n] = in_flow * (port_fracs[n.index] / in_avail)
                else:
                    flows[n] = np.zeros(ndim)

            elif isinstance(n, Converger):
                in_vecs = [flows[inp] for inp in n.inputs]
                in_totals = np.array([float(np.sum(v)) for v in in_vecs])
                cap = min(float(scalar_demand[n]), 1.0)
                if cap > _EPS and np.any(in_totals > _EPS):
                    fracs = converger_explicit(in_totals / cap) * cap
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
            break

        # --- Backward pass ---
        for unit in reversed(units):
            if isinstance(unit, Converger):
                in_vecs = [flows[inp] for inp in unit.inputs]
                in_totals = np.array([float(np.sum(v)) for v in in_vecs])
                cap = min(float(scalar_demand[unit]), 1.0)
                if cap > _EPS and np.any(in_totals > _EPS):
                    fracs = converger_explicit(in_totals / cap) * cap
                    for inp, frac in zip(unit.inputs, fracs):
                        scalar_demand[inp] = float(frac)
                else:
                    for inp in unit.inputs:
                        scalar_demand[inp] = 0.0

            elif isinstance(unit, Splitter):
                in_flow = flows[unit.inp]
                in_avail = float(np.sum(in_flow))
                out_caps = np.array([scalar_demand[p] for p in unit._ports])
                if in_avail > _EPS:
                    port_fracs = converger_explicit(out_caps / in_avail) * in_avail
                else:
                    port_fracs = np.zeros(unit.n)
                scalar_demand[unit.inp] = float(np.sum(port_fracs))

    return flows
