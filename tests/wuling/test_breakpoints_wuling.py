"""Regression tests: find_breakpoints reproduces historical hand-derived
breakpoints from the Wuling scenario tests, instead of requiring them to be
worked out by hand."""

from fractions import Fraction

from factorylib.breakpoints import find_breakpoints

from ._helpers import BASE_INCOME, _make_wuling_formulas, _search, make_formula
from .test_xiranite_gourd import _make_xg_formulas


def test_heavy_xiranite_worth_breakpoint():
    """Matches test_baseline.py::test_wuling_1p2_heavy_xiranite_worth's
    documented "$19: exact breakpoint" comment."""

    def g(hx_worth):
        f = _make_wuling_formulas()
        f["hx"] = make_formula([60 + 30 * 4 / 5, 0, 0, 0], output=hx_worth * 6)
        best, _, _ = _search(BASE_INCOME, f)
        return best.formula_rates[6]  # xi passthrough rate: jumps 134 -> 20 at $19

    bps = find_breakpoints(g, 10, 30, coarse_steps=60)
    assert len(bps) == 1
    assert bps[0].x_snapped == Fraction(19)


def test_xiranite_gourd_blue_cert_breakpoints():
    """Matches test_xiranite_gourd.py's documented bc breakpoints:
    19/8 (metatransfer flip), 303/64 (HC switches on), 5381/1024 (SC -> 0)."""

    def g(bc):
        f = _make_xg_formulas(bc)
        best, _, _ = _search(BASE_INCOME, f)
        return tuple(best.formula_rates)

    # Numerical search accumulates more floating-point noise from the LP
    # solver than snapping an already-computed exact value does, so a
    # looser snap_tol than the 1e-9 default is appropriate here. 5381/1024's
    # denominator also exceeds the default max_denom=1000, so raise it.
    bps = find_breakpoints(g, 0, 10, coarse_steps=100, snap_tol=1e-6, max_denom=2000)
    snapped = [bp.x_snapped for bp in bps]
    assert snapped == [Fraction(19, 8), Fraction(303, 64), Fraction(5381, 1024)]
