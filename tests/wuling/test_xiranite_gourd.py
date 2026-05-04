"""Wuling 1.2 "Palm-Top Savior" Event Phase 1 — Xiranite Gourd.

New formulas added to Wuling 1.2 base (order: sc lc hp hx ya yc xi cp hc xg):
  hc: HC Valley Battery [0, 180, 120, 0] → 54*6*1100/3200 $/run
      (shadow cost of OriOre+FerOre too high; rate=0 in all solutions)
  xg: Xiranite Gourd    [90, 0, 0, 0]   → (40 + bc*10)*6 $/run
      bc = $ value per blue cert (event currency)

Structural breakpoints:
  bc = 19/8      — metatransfer flips from [0,50,0,0] to [0,0,25,0]
  bc = 303/64    — HC Battery switches on; yc drops to 0; SC partial
  bc = 5381/1024 — SC drops to 0; OriOre freed (slack = 615/2)
"""
import numpy as np
import pytest

from ._helpers import BASE_INCOME, _make_wuling_formulas, _search, make_formula


def _make_xg_formulas(bc, purification=True):
    """Wuling 1.2 + HC Valley Battery + Xiranite Gourd at bc $ per blue cert."""
    f = _make_wuling_formulas(purification)
    f["hc"] = make_formula([0, 180, 120, 0], output=54 * 6 * 1100 / 3200)
    f["xg"] = make_formula([90, 0, 0, 0], output=(40 + bc * 10) * 6)
    return f


@pytest.mark.parametrize(
    "bc,expected_dollar,expected_rates,expected_slack,expected_z,expected_mt",
    [
        # 1 blue cert = $0 — XG worth $240/run; XG absorbs Xi freed by dropping hx
        (
            0,
            7739 / 6,
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 67 / 45],
            [0, 0, 0, 0],
            8,
            [0, 50, 0, 0],
        ),
        # 1 blue cert = $1 — XG worth $300/run; same allocation
        (
            1,
            8275 / 6,
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 67 / 45],
            [0, 0, 0, 0],
            8,
            [0, 50, 0, 0],
        ),
        # 1 blue cert = $10 — XG worth $840/run; SC fully displaced, OriOre slack
        (
            10,
            162863 / 64,
            [0, 0, 0, 0, 3 / 2, 0, 0, 0, 23 / 24, 8 / 3],
            [0, 615 / 2, 0, 0],
            8,
            [0, 0, 25, 0],
        ),
    ],
)
def test_wuling_1p2_xg_blue_cert_rate(
    bc, expected_dollar, expected_rates, expected_slack, expected_z, expected_mt
):
    f = _make_xg_formulas(bc)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, expected_dollar)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, expected_slack)
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)


# ---------------------------------------------------------------------------
# Blue cert value sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bc,expected_dollar,expected_rates,expected_z,expected_mt",
    [
        # $2/cert: below MT breakpoint ($19/8); metatransfer stays [0,50,0,0]
        (
            2,
            2937 / 2,
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 67 / 45],
            8,
            [0, 50, 0, 0],
        ),
        # $19/8/cert: exact MT breakpoint — both MTs yield $1502;
        #     tiebreak keeps [0,50,0,0]
        (
            19 / 8,
            1502,
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 67 / 45],
            8,
            [0, 50, 0, 0],
        ),
        # $3/cert: above MT breakpoint; metatransfer flips to [0,0,25,0]
        (3, 1562, [2, 0, 0, 0, 3 / 2, 11 / 24, 0, 0, 0, 8 / 5], 8, [0, 0, 25, 0]),
        # $4/cert: below HC breakpoint ($303/64); SC=2, yc active
        (4, 1658, [2, 0, 0, 0, 3 / 2, 11 / 24, 0, 0, 0, 8 / 5], 8, [0, 0, 25, 0]),
        # $303/64/cert: exact HC breakpoint — both strategies yield $3457/2;
        #     solver picks HC-on (yc drops to 0, SC partial)
        (
            303 / 64,
            3457 / 2,
            [41 / 26, 0, 0, 0, 3 / 2, 0, 0, 0, 22 / 39, 356 / 195],
            8,
            [0, 0, 25, 0],
        ),
        # $5/cert: above HC breakpoint, below SC=0 breakpoint ($5381/1024)
        (
            5,
            91395 / 52,
            [41 / 26, 0, 0, 0, 3 / 2, 0, 0, 0, 22 / 39, 356 / 195],
            8,
            [0, 0, 25, 0],
        ),
        # $5381/1024/cert: exact SC=0 breakpoint — both strategies yield $114273/64;
        #     solver keeps SC-partial allocation
        (
            5381 / 1024,
            114273 / 64,
            [41 / 26, 0, 0, 0, 3 / 2, 0, 0, 0, 22 / 39, 356 / 195],
            8,
            [0, 0, 25, 0],
        ),
        # $6/cert: above SC=0 breakpoint; SC fully displaced, OriOre freed
        (
            6,
            121903 / 64,
            [0, 0, 0, 0, 3 / 2, 0, 0, 0, 23 / 24, 8 / 3],
            8,
            [0, 0, 25, 0],
        ),
    ],
)
def test_wuling_1p2_xg_blue_cert_worth(
    bc, expected_dollar, expected_rates, expected_z, expected_mt
):
    f = _make_xg_formulas(bc)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, expected_dollar)
    assert np.allclose(best.formula_rates, expected_rates)
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
