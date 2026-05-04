"""Tests for Xiranite Jade Gourd scenario.

Uses intermediate variables (supply=0 + negative consumption):
  heavy_xi:  produced by hx_make, split between hx_sell and jg
  hetonite:  produced by hetonite_make, split between hp_sell and jg

Resources: [xi, ori_ore, ferrium, cuprium, heavy_xi, hetonite]
Intermediates (supply=0): heavy_xi (index 4), hetonite (index 5).

Formula order (indices 0-12):
  sc(0), lc(1), hx_make(2), hx_sell(3), ya(4), yc(5),
  xi_sell(6), cp_sell(7), hetonite_make(8), hp_sell(9), hc(10), xg(11), jg(12)

Breakpoints (bc=0):
  bp1_price = 904/15   -- jg_price where z=7 jg=1 ties z=8 jg=0
  bp2_price = 1279099/11520 -- jg_price where z=6 jg=2 ties z=7 jg=1

bc→∞ breakpoint:
  bc_inf_jg_bc = 38/3  -- jg_bc where xg and jg are equally profitable
    (derived from Δxi between z=8 and z=7: 114 xi, vs jg yield 6 per forge)
"""
import numpy as np
import pytest

from factorylib.optimize import Formula, maximize_dollar

BASE_INCOME = np.array([0, 480, 90, 180, 0, 0], dtype=float)
XI_PER_FORGE = np.array([30, 0, 0, 0, 0, 0], dtype=float)
METATRANSFERS = [[0, 50, 0, 0, 0, 0], [0, 0, 25, 0, 0, 0]]


def _f(consumption, output, limit=np.inf):
    return Formula(np.array(consumption, dtype=float), output, limit)


def _make_formulas(
    jg_price=0,
    bc=0,
    jg_bc=0,
    purification=True,
    include_hc=True,
    include_xg=True,
    include_jg=True,
    bc_only=False,
):
    """Build formula dict.

    bc_only=True: zero all dollar outputs; set bc=1 internally.
    Models the bc→∞ limit exactly — only cert-generating formulas matter.
    """
    xi_sc = 60 * (4 / 5 if purification else 1)
    xi_hx = 60 + 30 * (4 / 5 if purification else 1)
    ds = 0.0 if bc_only else 1.0      # dollar scale
    bc_eff = 1 if bc_only else bc     # cert value (1 cert = 1 unit in bc_only)
    f = {
        "sc":            _f([xi_sc, 240, 30, 0, 0, 0], ds * 54 * 6),
        "lc":            _f([30, 180, 0, 0, 0, 0],     ds * 25 * 6),
        "hx_make":       _f([xi_hx, 0, 0, 0, -6, 0],   0),
        "hx_sell":       _f([0, 0, 0, 0, 6, 0],        ds * 27 * 6),
        "ya":            _f([0, 0, 0, 120, 0, 0],       ds * 22 * 6),
        "yc":            _f([0, 0, 120, 0, 0, 0],       ds * 16 * 6),
        "xi_sell":       _f([1, 0, 0, 0, 0, 0],         ds),
        "cp_sell":       _f([0, 0, 0, 1, 0, 0],         ds),
        "hetonite_make": _f([0, 0, 30, 240, 0, -30],    0),
        "hp_sell":       _f([0, 0, 0, 0, 0, 30],        ds * 48 * 6),
    }
    if include_hc:
        f["hc"] = _f([0, 180, 120, 0, 0, 0], ds * 54 * 6 * 1100 / 3200)
    if include_xg:
        f["xg"] = _f([90, 0, 0, 0, 0, 0], (ds * 40 + bc_eff * 10) * 6)
    if include_jg:
        f["jg"] = _f([0, 0, 0, 0, 6, 6], (ds * jg_price + bc_eff * jg_bc) * 6)
    return f


def _search(base_income, formulas, max_forges=8, metatransfers=None):
    if metatransfers is None:
        metatransfers = METATRANSFERS
    candidates = []
    for z in range(max_forges + 1):
        formulas["hx_make"].limit = max_forges - z
        for mt in metatransfers:
            income = base_income + z * XI_PER_FORGE + np.array(mt, dtype=float)
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, list(mt)))
    return max(candidates, key=lambda r: r[0].dollar_output)


# ---------------------------------------------------------------------------
# Regression: new-model formulation is equivalent to test_wuling_1p2_full
# Splits hx → hx_make + hx_sell, hp → hetonite_make + hp_sell.
# Dollar output must match the original combined-formula result.
# ---------------------------------------------------------------------------


def test_wuling_new_model_equiv():
    f = _make_formulas(include_hc=False, include_xg=False, include_jg=False)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 2229 / 2)
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0, 0, 0])
    assert np.allclose(best.resource_slack[:4], [0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Goal 1: jade gourd sell price breakpoints (bc=0, jg_bc=0)
#
# bp1 = 904/15 ≈ 60.27: below → z=8 no jg; above → z=7 jg=1
# bp2 = 1279099/11520 ≈ 111.03: below → z=7 jg=1; above → z=6 jg=2
#
# Below bp1 (z=8):  sc=53/24, ya=3/2, yc=19/96, xg=67/45, jg=0
# Above bp1 (z=7):  sc=53/24, hx_make=1, ya=11/10, yc=71/480,
#                   hetonite_make=1/5, xg=2/9, jg=1  (same rates for both test pts)
# Above bp2 (z=6):  sc=1/4, hx_make=2, ya=7/10, hetonite_make=2/5,
#                   hc=191/240, jg=2  (ferrium metatransfer; ori slack=1107/4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "jg_price,expected_z,expected_mt,expected_rates,expected_slack,expected_dollar",
    [
        (
            60, 8, [0, 50, 0, 0, 0, 0],
            [53/24, 0, 0, 0, 3/2, 19/96, 0, 0, 0, 0, 0, 67/45, 0],
            [0, 0, 0, 0, 0, 0],
            7739/6,
        ),
        (
            61, 7, [0, 50, 0, 0, 0, 0],
            [53/24, 0, 1, 0, 11/10, 71/480, 0, 0, 1/5, 0, 0, 2/9, 1],
            [0, 0, 0, 0, 0, 0],
            38827/30,
        ),
        (
            111, 7, [0, 50, 0, 0, 0, 0],
            [53/24, 0, 1, 0, 11/10, 71/480, 0, 0, 1/5, 0, 0, 2/9, 1],
            [0, 0, 0, 0, 0, 0],
            47827/30,
        ),
        (
            112, 6, [0, 0, 25, 0, 0, 0],
            [1/4, 0, 2, 0, 7/10, 0, 0, 0, 2/5, 0, 191/240, 0, 2],
            [0, 1107/4, 0, 0, 0, 0],
            1027863/640,
        ),
    ],
)
def test_jade_gourd_price_breakpoints(
    jg_price, expected_z, expected_mt, expected_rates, expected_slack, expected_dollar
):
    f = _make_formulas(jg_price=jg_price)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)


# ---------------------------------------------------------------------------
# Goal 2: bc (bonus certificate) curve, jg_price fixed at bp1 = 904/15
#
# At bc→∞, the break-even jg_bc is 38/3 ≈ 12.67.
# Derivation: going z=8→z=7 gives 30 extra xi to xg but costs 1 jg.
#   z=7 uses 84 xi for hx_make; z=8 uses all 240 for xg.
#   Δxi(xg) = 240 - (210 - 84) = 114; Δjg = 1.
#   Break-even: 114/90 * 10bc*6 = jg_bc*bc*6 → jg_bc = 114*10/90 = 38/3.
# At bc→∞ above 38/3, z=6 jg=2 beats z=7 jg=1 (higher bc coefficient).
#
# bc_only=True: all $ outputs zeroed, bc_eff=1 — models bc→∞ exactly.
#   bc_only removes the $ correction that shifts metatransfer choice at finite bc.
#   Result: ori metatransfer wins (hc has 0 output, so ferrium MT adds nothing).
#   Dollar output is in cert units (clean integers: 160, 164).
#
# bc breakpoint for jg_bc=13 (> 38/3): ≈5.04 (z=7 jg=1 → z=6 jg=2)
# Below: z=7, mt=ori; above: z=6, mt=ferrium.
#
# bc=1, jg_bc=13 (z=7):  sc=53/24, hx_make=1, ya=11/10, yc=71/480,
#                         hetonite_make=1/5, xg=2/9, jg=1
# bc=6, jg_bc=13 (z=6):  hx_make=2, ya=7/10, hetonite_make=2/5,
#                         hc=103/120, xg=2/15, jg=2  (fer MT; ori slack=651/2)
# bc_only, jg_bc=12 (z=8): xg=8/3; all non-xi resources fully slack
# bc_only, jg_bc=13 (z=6): hx_make=2, hetonite_make∈[2/5,3/4] (degenerate),
#                           xg=2/15, jg=2  (cert output=164)
# ---------------------------------------------------------------------------

_BP1 = 904 / 15


@pytest.mark.parametrize(
    "bc,jg_bc,bc_only,expected_z,expected_mt,expected_rates,expected_slack,expected_dollar",
    [
        # jg_bc=13 > 38/3, bc=1: effective price 1099/15 between bp1 and bp2 → z=7 jg=1
        (
            1, 13, False, 7, [0, 50, 0, 0, 0, 0],
            [53/24, 0, 1, 0, 11/10, 71/480, 0, 0, 1/5, 0, 0, 2/9, 1],
            [0, 0, 0, 0, 0, 0],
            8287/6,
        ),
        # jg_bc=13, bc=6 > ~5.04 threshold → z=6 jg=2 (ferrium metatransfer)
        (
            6, 13, False, 6, [0, 0, 25, 0, 0, 0],
            [0, 0, 2, 0, 7/10, 0, 0, 0, 2/5, 0, 103/120, 2/15, 2],
            [0, 651/2, 0, 0, 0, 0],
            616703/320,
        ),
        # bc_only, jg_bc=12 < 38/3: xg (z=8) wins; cert output = 8/3*10*6 = 160
        (
            0, 12, True, 8, [0, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8/3, 0],
            [0, 530, 90, 180, 0, 0],
            160,
        ),
        # bc_only, jg_bc=13 > 38/3: jg (z=6) wins; cert output = 2/15*60 + 2*13*6 = 164
        # hetonite_make is degenerate: cuprium limits it to ≤3/4 but LP only needs ≥2/5.
        # expected_rates/slack are None; non-degenerate entries asserted separately below.
        (
            0, 13, True, 6, [0, 50, 0, 0, 0, 0],
            None,
            None,
            164,
        ),
    ],
)
def test_jade_gourd_bc_worth(
    bc, jg_bc, bc_only, expected_z, expected_mt, expected_rates, expected_slack, expected_dollar
):
    f = _make_formulas(jg_price=_BP1, bc=bc, jg_bc=jg_bc, bc_only=bc_only)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    fkeys = list(f)
    assert best.status == "optimal"
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
    if expected_rates is not None:
        assert np.allclose(best.formula_rates, expected_rates)
    else:
        assert np.isclose(best.formula_rates[fkeys.index("jg")], 2)
        assert np.isclose(best.formula_rates[fkeys.index("xg")], 2 / 15)
        hm = best.formula_rates[fkeys.index("hetonite_make")]
        assert 2/5 <= hm <= 3/4 + 1e-9
    if expected_slack is not None:
        assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)
