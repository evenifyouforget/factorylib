"""Tests for factorylib.diagram (wrap_label/build_graph/render_graph),
ported (and split out) from factorylib.endfield.main's monolithic
optimize()."""

from __future__ import annotations

import graphviz

from factorylib.diagram import build_graph, render_graph, wrap_label
from factorylib.fractions import snap_multiples
from factorylib.material import HIDDEN, SOLID, VIRTUAL, Material, Recipe
from factorylib.optimize import solve


def test_wrap_label_wraps_long_lines():
    text = "a " * 30
    wrapped = wrap_label(text)
    assert all(len(line) <= 40 for line in wrapped.splitlines())


def test_wrap_label_preserves_existing_newlines():
    text = "first line\nsecond line"
    wrapped = wrap_label(text)
    assert wrapped.splitlines()[0] == "first line"
    assert wrapped.splitlines()[1] == "second line"


def _small_result():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    result = solve(set(), [supply, sell], dollar)
    multiples = snap_multiples(result.multiples)
    return result, multiples


def test_build_graph_returns_digraph():
    result, multiples = _small_result()
    dot = build_graph(result, multiples)
    assert isinstance(dot, graphviz.Digraph)


def test_build_graph_hides_zero_activity_material():
    """A material that's never produced or consumed at these multiples gets
    no node (it would otherwise clutter the diagram with a dead 0/0 node)."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    unused = Material(name="Unused", unit="/min", tags=SOLID)
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    stub = Recipe(0 * unused, name="Stub", max_multiples=0)
    result = solve(set(), [supply, sell, stub], dollar)
    multiples = snap_multiples(result.multiples)
    dot = build_graph(result, multiples)
    source = dot.source
    assert "Unused" not in source


def test_build_graph_hides_hidden_tagged_material():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    hidden_counter = Material(name="HiddenCounter", tags=VIRTUAL + HIDDEN)
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    assign = Recipe(-1 * hidden_counter + 1 * ore, name="Assign", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    result = solve({hidden_counter}, [supply, sell, assign], dollar)
    multiples = snap_multiples(result.multiples)
    dot = build_graph(result, multiples)
    assert "HiddenCounter" not in dot.source


def test_render_graph_writes_file(tmp_path):
    result, multiples = _small_result()
    dot = build_graph(result, multiples)
    outfile = tmp_path / "graph"
    render_graph(dot, str(outfile))
    assert outfile.exists()
