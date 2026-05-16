"""Wuling 1.2e scenario tests.

Built off 1.2d. Changes vs 1.2d:
  - Cuprium Ore is now the base resource (was Cuprium).
  - cup_conv:    30 Cuprium Ore → 30 Cuprium + 30 Sewage           [new]
  - xi_sew:      30 Xiranite + 30 Sewage → 30 Effluent + 30 Inert  [new]
  - SC path:     60 Eff + 30 Ferr → 30 Xircon + 30 Sew;
                 30 Xircon + 240 Ori → 6 SC                         [refactored]
  - HP path:     240 Cup + 30 Ferr → 30 Hetonite + 30 Sew;
                 30 Hetonite → 6 HP                                  [refactored]
  - HX:          60 Xi + 30 Eff → 6 HX                              [refactored]
  - purify:      Purification Building: 120 Inert Eff → 30 Eff
  - purify_node: Test Area Purification Node: 30 Sew → 1 Eff
                 (max 12 multiples; creates sewage↔effluent cycle)   [new]

Resource vector (8): [xi, ori, ferr, cup_ore, cup, sew, eff, inert]

Formula order: cup_conv  xi_sew  sc  lc  hp  hx  ya  yc  xi_sell  cp_sell
               [purify]  [purify_node]

Equivalence:
  no purify_node + 1.2d limits  → dollar = 2823/2  (same as 1.2d baseline)
  no purify_node + 1.2 full limits → dollar = 2229/2 (same as 1.2 full baseline)

Full 1.2e (both purify_building + purify_node, 1.2d limits):
  z=10, mt=ori, dollar=206735/146 ≈ 1416.0  (vs 1.2d 2823/2 = 1411.5)
"""

import numpy as np
import pytest

from factorylib.optimize import maximize_dollar

from ._helpers import make_formula

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_1P2E = np.array([0, 540, 90, 240, 0, 0, 0, 0], dtype=float)
BASE_1P2_FULL = np.array([0, 480, 90, 180, 0, 0, 0, 0], dtype=float)
MAX_FORGES_1P2E = 12
MAX_FORGES_1P2_FULL = 8

_XI8 = np.array([30, 0, 0, 0, 0, 0, 0, 0], dtype=float)
_MTS8 = [
    [0, 50, 0, 0, 0, 0, 0, 0],
    [0, 0, 25, 0, 0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# Formula factory
# ---------------------------------------------------------------------------


def _make_1p2e_formulas(purify_node=True, purify_building=True):
    """Return fresh dict of 1.2e formulas.

    Resources: [xi, ori, ferr, cup_ore, cup, sew, eff, inert]
    Negatives in consumption = production (same convention as 5R effluent model).
    Xircon and Hetonite are intermediate-collapsed into sc and hp respectively.
    """
    f = {
        # 30 cup_ore → 30 cup + 30 sew
        "cup_conv": make_formula([0, 0, 0, 30, -30, -30, 0, 0], output=0),
        # 30 xi + 30 sew → 30 eff + 30 inert
        "xi_sew": make_formula([30, 0, 0, 0, 0, 30, -30, -30], output=0),
        # SC: (60 eff + 30 ferr → 30 Xircon + 30 sew) + (30 Xircon + 240 ori → 6 SC)
        "sc": make_formula([0, 240, 30, 0, 0, -30, 60, 0], output=54 * 6),
        # LC: unchanged — 30 xi + 180 ori → 6 LC
        "lc": make_formula([30, 180, 0, 0, 0, 0, 0, 0], output=25 * 6),
        # HP: (240 cup + 30 ferr → 30 Hetonite + 30 sew) + (30 Hetonite → 6 HP)
        "hp": make_formula([0, 0, 30, 0, 240, -30, 0, 0], output=48 * 6),
        # HX: 60 xi + 30 eff → 6 HX (forge-limited; limit set by search)
        "hx": make_formula([60, 0, 0, 0, 0, 0, 30, 0], output=27 * 6),
        # YA: 120 cup → 6 ya
        "ya": make_formula([0, 0, 0, 0, 120, 0, 0, 0], output=22 * 6),
        # YC: 120 ferr → 6 yc
        "yc": make_formula([0, 0, 120, 0, 0, 0, 0, 0], output=16 * 6),
        # Sell xi at $1
        "xi_sell": make_formula([1, 0, 0, 0, 0, 0, 0, 0], output=1),
        # Sell cup at $1 (after cup_conv conversion)
        "cp_sell": make_formula([0, 0, 0, 0, 1, 0, 0, 0], output=1),
    }
    if purify_building:
        # Purification Building: 120 inert → 30 eff
        f["purify"] = make_formula([0, 0, 0, 0, 0, 0, -30, 120], output=0)
    if purify_node:
        # Test Area Purification Node: 30 sew → 1 eff (max 12 multiples)
        f["purify_node"] = make_formula([0, 0, 0, 0, 0, 30, -1, 0], output=0, limit=12)
    return f


def _search_1p2e(formulas, base=BASE_1P2E, max_forges=MAX_FORGES_1P2E):
    candidates = []
    for z in range(max_forges + 1):
        formulas["hx"].limit = max_forges - z
        for mt in _MTS8:
            income = base + z * _XI8 + np.array(mt, dtype=float)
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, list(mt[:4])))
    return max(candidates, key=lambda r: r[0].dollar_output)


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------


def test_1p2e_equiv_1p2d():
    """Without purify_node, 1.2d limits → same result as 1.2d baseline.

    Formula order: cup_conv xi_sew sc lc hp hx ya yc xi_sell cp_sell purify
    Sewage accumulates as slack (produced by cup_conv/sc/hp; only partially
    consumed by xi_sew which needs xi as co-reactant).
    """
    f = _make_1p2e_formulas(purify_node=False)
    best, best_z, best_mt = _search_1p2e(f)
    assert best.status == "optimal"
    assert best_z == 10
    assert np.allclose(best_mt, [0, 50, 0, 0])
    assert np.allclose(
        best.formula_rates,
        [8, 83 / 15, 59 / 24, 0, 13 / 24, 2, 11 / 12, 0, 14, 0, 83 / 60],
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0, 0, 164, 0, 0])
    assert np.isclose(best.dollar_output, 2823 / 2)


def test_1p2e_equiv_1p2_full():
    """Without purify_node, 1.2 full limits → same dollar as 1.2 full baseline.

    Formula order: cup_conv xi_sew sc lc hp hx ya yc xi_sell cp_sell purify
    """
    f = _make_1p2e_formulas(purify_node=False)
    best, best_z, best_mt = _search_1p2e(
        f, base=BASE_1P2_FULL, max_forges=MAX_FORGES_1P2_FULL
    )
    assert best.status == "optimal"
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])
    assert np.allclose(
        best.formula_rates,
        [6, 13 / 3, 53 / 24, 0, 3 / 4, 1, 0, 1 / 96, 20, 0, 13 / 12],
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0, 0, 555 / 4, 0, 0])
    assert np.isclose(best.dollar_output, 2229 / 2)


# ---------------------------------------------------------------------------
# Full 1.2e baseline (purify_building + purify_node)
# ---------------------------------------------------------------------------


def test_1p2e_full():
    """Full 1.2e model — purify_node adds ~4.5 $/min vs 1.2d.

    Formula order: cup_conv xi_sew sc lc hp hx ya yc xi_sell cp_sell purify purify_node

    Key rates (exact fractions, all slacks 0):
      xi_sew      = 393/73   purify      = 393/292 = xi_sew/4 (inert balance)
      purify_node = 410/73   xi_sell     = 1350/73
    """
    f = _make_1p2e_formulas()
    best, best_z, best_mt = _search_1p2e(f)
    assert best.status == "optimal"
    assert best_z == 10
    assert np.allclose(best_mt, [0, 50, 0, 0])
    assert np.allclose(
        best.formula_rates,
        [
            8,
            393 / 73,
            59 / 24,
            0,
            13 / 24,
            2,
            11 / 12,
            0,
            1350 / 73,
            0,
            393 / 292,
            410 / 73,
        ],
    )
    assert np.allclose(best.resource_slack, [0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(best.dollar_output, 206735 / 146)


# ---------------------------------------------------------------------------
# Variants (parametrized)
#
# All use full 1.2e (purify_building + purify_node) + 1.2d limits.
#
# ban_ya:            ya=0; freed cup → cp_sell; dollar drops ~11 $/min
# ban_yc:            yc=0; yc was already 0 in optimal → no change
# ban_hp:            hp=0; cup freed; cup+sew cycle adjusts xi_sew/purify;
#                    dollar halves (hp was major contributor)
# sc_cap_2:          sc≤2; lc fills freed ori; dollar drops ~4 $/min
# no_purify_building:purify=0; inert accumulates as slack; xi unavailable
#                    for xi_sell (all goes to xi_sew); dollar drops ~59 $/min
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario,setup,expected_z,expected_mt,expected_rates,expected_slack,expected_dollar",
    [
        (
            "ban_ya",
            {"ya": 0},
            10,
            [0, 50, 0, 0],
            # cup freed from ya (240/min) → cp_sell; sewage/eff cycle unchanged
            [
                8,
                393 / 73,
                59 / 24,
                0,
                13 / 24,
                2,
                0,
                0,
                1350 / 73,
                110,
                393 / 292,
                410 / 73,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0],
            205129 / 146,
        ),
        (
            "ban_yc",
            {"yc": 0},
            10,
            [0, 50, 0, 0],
            # yc=0 in unconstrained optimal → no change
            [
                8,
                393 / 73,
                59 / 24,
                0,
                13 / 24,
                2,
                11 / 12,
                0,
                1350 / 73,
                0,
                393 / 292,
                410 / 73,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0],
            206735 / 146,
        ),
        (
            "ban_hp",
            {"hp": 0},
            10,
            [0, 50, 0, 0],
            # hp=0; 240 sew/min from cup_conv + 73.75/min from sc only
            # purify = xi_sew/4 (exact: 4729/3504, denom > 1000 so snap_value warned)
            [
                8,
                4729 / 876,
                59 / 24,
                0,
                0,
                2,
                2,
                13 / 96,
                2635 / 146,
                0,
                4729 / 3504,
                2955 / 584,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0],
            103335 / 73,
        ),
        (
            "sc_cap_2",
            {"sc": 2},
            10,
            [0, 50, 0, 0],
            [8, 338 / 73, 2, 11 / 18, 1, 2, 0, 0, 4985 / 219, 0, 169 / 146, 465 / 73],
            [0, 0, 0, 0, 0, 0, 0, 0],
            301000 / 219,
        ),
        (
            "no_purify_building",
            {"purify": 0},
            10,
            [0, 50, 0, 0],
            # purify=0; inert from xi_sew accumulates; all xi consumed by xi_sew+hx
            [
                8,
                573 / 88,
                59 / 24,
                0,
                13 / 24,
                307 / 176,
                11 / 12,
                0,
                0,
                0,
                0,
                395 / 88,
            ],
            [0, 0, 0, 0, 0, 0, 0, 8595 / 44],
            119335 / 88,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_1p2e_variants(
    scenario,
    setup,
    expected_z,
    expected_mt,
    expected_rates,
    expected_slack,
    expected_dollar,
):
    f = _make_1p2e_formulas()
    for key, val in setup.items():
        f[key].limit = val
    best, best_z, best_mt = _search_1p2e(f)
    assert best.status == "optimal"
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)
