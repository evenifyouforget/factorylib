import numpy as np
import pytest

from factorylib.network import Converger, Source, Splitter, solve


def test_source():
    a = Source([1, 0])
    flows = solve(a)
    assert np.allclose(flows[a], [1, 0])


def test_converger_equal_sources():
    a = Source([1, 0])
    b = Source([0, 1])
    c = Converger([a, b])
    flows = solve(c)
    assert np.allclose(flows[c], [0.5, 0.5])


def test_converger_one_empty():
    a = Source([1, 0])
    b = Source([0, 0])
    c = Converger([a, b])
    flows = solve(c)
    assert np.allclose(flows[c], [1, 0])


def test_converger_unsaturated():
    a = Source([0.3, 0])
    b = Source([0, 0.4])
    c = Converger([a, b])
    flows = solve(c)
    assert np.allclose(flows[c], [0.3, 0.4])


def test_splitter_even():
    a = Source([1, 0])
    e = Splitter(a, n=2)
    flows = solve(e[0])
    assert np.allclose(flows[e[0]], [0.5, 0])
    assert np.allclose(flows[e[1]], [0.5, 0])


def test_splitter_three():
    a = Source([1, 0])
    e = Splitter(a, n=3)
    flows = solve(e[0])
    assert np.allclose(flows[e[0]], [1 / 3, 0])
    assert np.allclose(flows[e[1]], [1 / 3, 0])
    assert np.allclose(flows[e[2]], [1 / 3, 0])


def test_dag_nontrivial():
    # A -> E (splitter) -> e[0] -> C, e[1] -> D
    # B -> C
    # C -> D -> output
    # Expected: D = 3/4 A + 1/4 B; A stalls at 3/4, B stalls at 1/4
    a = Source([1, 0])
    b = Source([0, 1])
    e = Splitter(a, n=2)
    c = Converger([b, e[0]])
    d = Converger([c, e[1]])
    flows = solve(d)
    assert np.allclose(flows[d], [0.75, 0.25])
    assert np.allclose(flows[a], [0.75, 0])
    assert np.allclose(flows[b], [0, 0.25])


def test_stall_source():
    a = Source([1, 0])
    b = Source([1, 0])
    c = Converger([a, b])
    flows = solve(c)
    assert np.allclose(flows[c], [1, 0])
    assert np.allclose(flows[a], [0.5, 0])
    assert np.allclose(flows[b], [0.5, 0])


@pytest.mark.parametrize("a_flow", [0, 0.1, 0.5, 0.9, 1])
def test_split_converge(a_flow):
    a = Source([a_flow])
    b = Splitter(a, n=2)
    c = Converger([b[0], b[1]])
    flows = solve(c)
    assert np.allclose(flows[c], [a_flow])
    assert np.allclose(flows[b[0]], [a_flow / 2])
    assert np.allclose(flows[b[1]], [a_flow / 2])


@pytest.mark.parametrize("a_flow", [0, 0.1, 0.5, 0.9, 1])
def test_zipper(a_flow):
    a = Source([a_flow])
    b = Splitter(a, n=2)
    c = Splitter(b[0], n=2)
    d = Converger([b[1], c[0]])
    e = Converger([c[1], d])
    flows = solve(e)
    assert np.allclose(flows[e], [a_flow])


@pytest.mark.parametrize("a_flow", [0, 0.1, 0.5, 0.9, 1])
def test_reverse_zipper(a_flow):
    a = Source([a_flow])
    b = Splitter(a, n=2)
    c = Splitter(b[0], n=2)
    d = Converger([c[1], c[0]])
    e = Converger([b[1], d])
    flows = solve(e)
    assert np.allclose(flows[e], [a_flow])
