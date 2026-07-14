from fractions import Fraction

from factorylib.breakpoints import find_breakpoints, max_abs_diff


def test_max_abs_diff_scalars():
    assert max_abs_diff(1.0, 3.0) == 2.0


def test_max_abs_diff_vectors():
    assert max_abs_diff((0.0, 0.0), (1.0, -2.0)) == 2.0


def test_single_step():
    def g(x):
        return 0.0 if x < 2.5 else 1.0

    bps = find_breakpoints(g, 0, 5)
    assert len(bps) == 1
    assert bps[0].x_snapped == Fraction(5, 2)
    assert bps[0].value_below == 0.0
    assert bps[0].value_above == 1.0


def test_multi_step():
    def g(x):
        return float(x >= 1) + float(x >= 3) + float(x >= 7)

    bps = find_breakpoints(g, 0, 10, coarse_steps=400)
    assert [bp.x_snapped for bp in bps] == [Fraction(1), Fraction(3), Fraction(7)]


def test_array_valued_step():
    def g(x):
        return (0.0, 0.0) if x < 1 / 3 else (1.0, -1.0)

    bps = find_breakpoints(g, 0, 1)
    assert len(bps) == 1
    assert bps[0].x_snapped == Fraction(1, 3)


def test_no_step_returns_empty():
    def g(x):
        return 2.0 * x

    assert find_breakpoints(g, 0, 10) == []


def test_steep_but_continuous_kink_is_not_a_breakpoint():
    """A steep-but-continuous kink should be rejected: the jump across a
    shrinking bracket around it decays toward 0, unlike a genuine jump."""

    def g(x):
        return x if x < 1 else 1 + 100 * (x - 1)

    bps = find_breakpoints(g, 0, 2, coarse_steps=20, min_jump=1e-6, x_tol=1e-9)
    assert bps == []


def test_raw_float_returned_when_not_a_simple_fraction():
    import math

    def g(x):
        return 0.0 if x < math.pi else 1.0

    bps = find_breakpoints(g, 0, 5, max_denom=10)
    assert len(bps) == 1
    assert isinstance(bps[0].x_snapped, float)
