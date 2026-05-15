"""Wuling 1.2 baseline scenario tests.

8 Forge of the Sky; Z forges → Xiranite supply, 8-Z → Heavy Xiranite cap.
Base supply: 0 Xiranite (+30*Z), 480 Originium Ore, 90 Ferrium Ore, 180 Cuprium Ore.
Metatransfer choice: 25 DOP (= 50 Originium Ore) or 25 Ferrium Ore.
"""

import numpy as np
import pytest

from factorylib.optimize import maximize_dollar

from ._helpers import (
    BASE_INCOME,
    METATRANSFERS,
    _make_wuling_formulas,
    _search,
    make_formula,
)


def test_wuling_1p2_full():
    f = _make_wuling_formulas()
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 2229 / 2)
    assert np.allclose(best.formula_rates, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])


# ---------------------------------------------------------------------------
# Fixed Z
# ---------------------------------------------------------------------------


def test_wuling_1p2_z6_fixed():
    f = _make_wuling_formulas()
    f["hx"].limit = 2
    cands = []
    for mt in METATRANSFERS:
        income = BASE_INCOME + 6 * np.array([30, 0, 0, 0], dtype=float) + mt
        cands.append((maximize_dollar(income, list(f.values())), mt))
    best, best_mt = max(cands, key=lambda x: x[0].dollar_output)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 15053 / 14)
    assert np.allclose(
        best.formula_rates, [53 / 24, 0, 0, 37 / 42, 3 / 2, 19 / 96, 0, 0]
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_z7_fixed():
    f = _make_wuling_formulas()
    f["hx"].limit = 1
    cands = []
    for mt in METATRANSFERS:
        income = BASE_INCOME + 7 * np.array([30, 0, 0, 0], dtype=float) + mt
        cands.append((maximize_dollar(income, list(f.values())), mt))
    best, best_mt = max(cands, key=lambda x: x[0].dollar_output)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 2229 / 2)
    assert np.allclose(best.formula_rates, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_z8_fixed():
    # All 8 forges on Xiranite supply; Heavy Xiranite has no capacity
    f = _make_wuling_formulas()
    f["hx"].limit = 0
    cands = []
    for mt in METATRANSFERS:
        income = BASE_INCOME + 8 * np.array([30, 0, 0, 0], dtype=float) + mt
        cands.append((maximize_dollar(income, list(f.values())), mt))
    best, best_mt = max(cands, key=lambda x: x[0].dollar_output)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1066.5)
    assert np.allclose(best.formula_rates, [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 134, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert np.allclose(best_mt, [0, 50, 0, 0])


# ---------------------------------------------------------------------------
# Restricted formula sets
# ---------------------------------------------------------------------------


def test_wuling_1p2_hx_lc_ya_only():
    # Only Heavy Xiranite, LC Wuling Battery, Yazhen Syringe A sellable
    f = _make_wuling_formulas()
    fonly = {k: f[k] for k in ["lc", "hx", "ya"]}
    best, best_z, best_mt = _search(BASE_INCOME, fonly)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 34291 / 42)
    assert np.allclose(best.formula_rates, [53 / 18, 275 / 252, 3 / 2])
    assert np.allclose(best.resource_slack, [0, 0, 90, 0])
    assert best_z == 6
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_sc_capped_2():
    # SC Wuling Battery limited to 2 multiples
    f = _make_wuling_formulas()
    f["sc"].limit = 2
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 3286 / 3)
    assert np.allclose(best.formula_rates, [2, 5 / 18, 0, 1, 3 / 2, 1 / 4, 65 / 3, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_sc_capped_2_no_lc():
    # SC capped at 2 multiples; LC Wuling Battery banned
    f = _make_wuling_formulas()
    f["sc"].limit = 2
    f["lc"].limit = 0
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1082)
    assert np.allclose(best.formula_rates, [2, 0, 0, 1, 3 / 2, 11 / 24, 30, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 0, 25, 0])


def test_wuling_1p2_sc_capped_2_no_lc_hc():
    # SC capped at 2; LC banned; HC Valley Battery available (180 Ori, 120 Fer)
    # HC Battery not used — shadow price of Originium+Ferrium too high vs output
    f = _make_wuling_formulas()
    f["sc"].limit = 2
    f["lc"].limit = 0
    f["hc"] = make_formula([0, 180, 120, 0], output=54 * 6 * 1100 / 3200)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1082)
    assert np.allclose(best.formula_rates, [2, 0, 0, 1, 3 / 2, 11 / 24, 30, 0, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 0, 25, 0])


def test_wuling_1p2_sc_capped_2_no_lc_no_mt():
    # SC capped at 2 multiples; LC banned; no metatransfer
    f = _make_wuling_formulas()
    f["sc"].limit = 2
    f["lc"].limit = 0
    best, best_z, best_mt = _search(BASE_INCOME, f, metatransfers=[[0, 0, 0, 0]])
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1062)
    assert np.allclose(best.formula_rates, [2, 0, 0, 1, 3 / 2, 1 / 4, 30, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 0, 0, 0])


def test_wuling_1p2_hx_uncapped():
    # Heavy Xiranite unlimited — forge constraint removed
    f = _make_wuling_formulas()
    f["hx"].limit = np.inf
    best, best_z, best_mt = _search(BASE_INCOME, f, fix_hx_limit=True)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 16673 / 14)
    assert np.allclose(
        best.formula_rates, [53 / 24, 0, 0, 67 / 42, 3 / 2, 19 / 96, 0, 0]
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 8
    assert np.allclose(best_mt, [0, 50, 0, 0])


# ---------------------------------------------------------------------------
# Each formula disabled (limit=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disabled,expected_dollar,expected_rates,expected_slack,expected_z,expected_mt",
    [
        # sc: LC Battery picks up Originium Ore;
        #     Xiranite passthrough absorbs freed Xiranite
        (
            "sc",
            2734 / 3,
            [0, 53 / 18, 0, 1, 3 / 2, 3 / 4, 113 / 3, 0],
            [0, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
        # lc: unused in full solution; no change
        (
            "lc",
            2229 / 2,
            [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0],
            [0, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
        # hp: unused in full solution; no change
        (
            "hp",
            2229 / 2,
            [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0],
            [0, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
        # hx: all forges → Xiranite supply; Xiranite passthrough absorbs excess
        (
            "hx",
            2133 / 2,
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 134, 0],
            [0, 0, 0, 0],
            8,
            [0, 50, 0, 0],
        ),
        # ya: Hetonite Part covers Cuprium Ore instead;
        #     Yazhen Syringe C handles Ferrium remainder
        (
            "ya",
            2229 / 2,
            [53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0],
            [0, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
        # yc: Ferrium Ore partially unspent (1.25/min slack)
        (
            "yc",
            2227 / 2,
            [53 / 24, 0, 3 / 4, 1, 0, 0, 20, 0],
            [0, 0, 5 / 4, 0],
            7,
            [0, 50, 0, 0],
        ),
        # xi: absorbs 20 Xi surplus in full solution;
        #     without it, 20 Xi wasted (slack)
        (
            "xi",
            2189 / 2,
            [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 0, 0],
            [20, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
        # cp: Cuprium Part passthrough unused in full solution; no change
        (
            "cp",
            2229 / 2,
            [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0],
            [0, 0, 0, 0],
            7,
            [0, 50, 0, 0],
        ),
    ],
)
def test_wuling_1p2_formula_disabled(
    disabled, expected_dollar, expected_rates, expected_slack, expected_z, expected_mt
):
    f = _make_wuling_formulas()
    f[disabled].limit = 0.0
    best, best_z, best_mt = _search(BASE_INCOME, f, fix_hx_limit=(disabled == "hx"))
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, expected_dollar)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, expected_slack)
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)


# ---------------------------------------------------------------------------
# Hetonite Part value sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hp_per_item,expected_dollar,expected_rates,expected_z,expected_mt",
    [
        # $24: well below breakpoint; never used
        (24, 2229 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $48: exact breakpoint — shadow cost of 30 FerOre + 240 CupOre equals output;
        #      not used (LP tiebreaks toward zero)
        (48, 2229 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $49: $1 above breakpoint; Hetonite Part switches on,
        #      displacing Yazhen Syringe A on Cuprium Ore
        (49, 1119, [53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $60: well above breakpoint
        (60, 2337 / 2, [53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $80: strongly prefer Hetonite Part
        (80, 2517 / 2, [53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0], 7, [0, 50, 0, 0]),
    ],
)
def test_wuling_1p2_hetonite_worth(
    hp_per_item, expected_dollar, expected_rates, expected_z, expected_mt
):
    f = _make_wuling_formulas()
    f["hp"] = make_formula([0, 0, 30, 240], output=hp_per_item * 6)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, expected_dollar)
    assert np.allclose(best.formula_rates, expected_rates)
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)


# ---------------------------------------------------------------------------
# Heavy Xiranite value sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hx_per_item,expected_dollar,expected_rates,expected_z,expected_mt",
    [
        # $18: below breakpoint ($19); Z=8 with Xiranite passthrough preferred
        (18, 2133 / 2, [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 134, 0], 8, [0, 50, 0, 0]),
        # $19: exact breakpoint — Z=7 and Z=8 yield equal dollar output;
        #      solver picks Z=7
        (19, 2133 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $20: $1 above breakpoint; Heavy Xiranite switches on with Z=7 allocation
        (20, 2145 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $27: base case (well above breakpoint)
        (27, 2229 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
        # $54: same allocation; high value
        (54, 2553 / 2, [53 / 24, 0, 0, 1, 3 / 2, 19 / 96, 20, 0], 7, [0, 50, 0, 0]),
    ],
)
def test_wuling_1p2_heavy_xiranite_worth(
    hx_per_item, expected_dollar, expected_rates, expected_z, expected_mt
):
    f = _make_wuling_formulas()
    f["hx"] = make_formula([60 + 30 * 4 / 5, 0, 0, 0], output=hx_per_item * 6)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, expected_dollar)
    assert np.allclose(best.formula_rates, expected_rates)
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)


# ---------------------------------------------------------------------------
# Supply / forge count variations
# ---------------------------------------------------------------------------


def test_wuling_1p2_7_forges():
    # Only 7 Forge of the Sky available
    f = _make_wuling_formulas()
    best, best_z, best_mt = _search(BASE_INCOME, f, max_forges=7)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 15053 / 14)
    assert np.allclose(
        best.formula_rates, [53 / 24, 0, 0, 37 / 42, 3 / 2, 19 / 96, 0, 0]
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 6
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_more_cuprium():
    # 240 Cuprium Ore available instead of 180
    f = _make_wuling_formulas()
    bi = BASE_INCOME.copy()
    bi[3] = 240
    best, best_z, best_mt = _search(bi, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 2361 / 2)
    assert np.allclose(best.formula_rates, [53 / 24, 0, 0, 1, 2, 19 / 96, 20, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])


def test_wuling_1p2_no_purification():
    # Purification Building not used: no 4/5 reduction on Xiranite cost
    # SC Wuling Battery: 60 Xi (up from 48); Heavy Xiranite: 90 Xi (up from 84)
    f = _make_wuling_formulas(purification=False)
    best, best_z, best_mt = _search(BASE_INCOME, f)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1072)
    assert np.allclose(
        best.formula_rates, [53 / 24, 0, 0, 31 / 36, 3 / 2, 19 / 96, 0, 0]
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])
