import numpy as np
import pytest

from factorylib.network import Converger, Source, Splitter, solve


def _half_rate_limiter(inp):
    """Source → Converger ↔ Splitter feedback loop. Output = min(inp_flow, 0.5)."""
    cvg = Converger([inp, None])
    spl = Splitter(cvg, n=2)
    cvg.set_input(1, spl[1])
    return spl[0]


def _fifth_splitter(inp):
    """1/5 virtual splitter via two nested splitters. Output = min(inp_flow/5, 1/6)."""
    c1 = Converger([inp, None])
    s_bot = Splitter(c1, weights=[1 / 3, 2 / 3])
    s_top = Splitter(s_bot[0], n=2)
    c1.set_input(1, s_top[0])
    return s_top[1]


_XS = [i / 10 for i in range(11)]
_AS = [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6]


@pytest.mark.parametrize("x", _XS)
def test_half_rate_limiter(x):
    """Example 1: Source → Converger → Splitter with feedback. Output = min(x, 0.5)."""
    src = Source([x])
    sink = _half_rate_limiter(src)
    result = solve(sink)
    assert np.allclose(result[sink], [min(x, 0.5)])
    assert result.converged


@pytest.mark.parametrize("x", _XS)
@pytest.mark.parametrize("a", _AS)
def test_arbitrary_rate_limiter(x, a):
    """Example 2: Weighted splitter feedback loop. Output = min(a, x)."""
    src = Source([x])
    cvg = Converger([src, None])
    spl = Splitter(cvg, weights=[a, 1 - a])
    cvg.set_input(1, spl[1])
    result = solve(spl[0])
    assert np.allclose(result[spl[0]], [min(a, x)])
    assert result.converged


@pytest.mark.parametrize("x", _XS)
def test_fifth_splitter(x):
    """Example 3: 1/5 virtual splitter via two nested cycles. Output = min(x/5, 1/6)."""
    src = Source([x])
    sink = _fifth_splitter(src)
    result = solve(sink)
    assert np.allclose(result[sink], [min(x / 5, 1 / 6)])
    assert result.converged


@pytest.mark.parametrize("x", _XS)
def test_composition_rate_limiter_then_fifth(x):
    """Example 4: Source → half_rate_limiter → fifth_splitter → Sink.
    Output = min(x, 0.5) / 5."""
    src = Source([x])
    rl_out = _half_rate_limiter(src)
    sink = _fifth_splitter(rl_out)
    result = solve(sink)
    assert np.allclose(result[sink], [min(x, 0.5) / 5])
    assert result.converged


@pytest.mark.parametrize("x", _XS)
def test_parallel_rate_limiters_nop(x):
    """Example 5: Two parallel half-rate-limiters recombined. Output = x (nop)."""
    src = Source([x])
    s = Splitter(src, n=2)
    out_a = _half_rate_limiter(s[0])
    out_b = _half_rate_limiter(s[1])
    c = Converger([out_a, out_b])
    result = solve(c)
    assert np.allclose(result[c], [x])
    assert result.converged
