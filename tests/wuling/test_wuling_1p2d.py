"""Wuling 1.2d scenario tests.

Built off 1.2 full (not 1.2c/jade-gourd model). Changes vs 1.2 full:
  - Originium Ore supply:  480 → 540/min
  - Cuprium Ore supply:    180 → 240/min
  - Forge of the Sky:        8 → 12 (split between Xi production and HX cap)

Baseline (4-resource [xi, ori, ferr, cup]):
  z=10, mt=ori, dollar=2823/2
  rates=[59/24, 0, 0, 2, 2, 13/96, 14, 0]  (sc lc hp hx ya yc xi cp)

Effluent model (5-resource [xi, ori, ferr, cup, eff]):
  Purification: 24 xi → 30 eff  (purify formula)
  sc consumes [0, 240, 30, 0, 60];  hx consumes [60, 0, 0, 0, 30]
  Formula order: purify sc lc hp hx ya yc xi cp
  eff=0 reproduces the 4-resource baseline exactly.
"""

import numpy as np
import pytest

from factorylib.optimize import maximize_dollar

from ._helpers import (
    METATRANSFERS,
    _make_wuling_formulas,
    _search,
    make_formula,
    snap_result,
)

BASE_1P2D = np.array([0, 540, 90, 240], dtype=float)
MAX_FORGES_1P2D = 12


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_wuling_1p2d_full():
    f = _make_wuling_formulas()
    best, best_z, best_mt = _search(BASE_1P2D, f, max_forges=MAX_FORGES_1P2D)
    assert best.status == "optimal"
    assert best_z == 10
    assert np.allclose(best_mt, [0, 50, 0, 0])
    assert np.allclose(best.formula_rates, [59 / 24, 0, 0, 2, 2, 13 / 96, 14, 0])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert np.isclose(best.dollar_output, 2823 / 2)


# ---------------------------------------------------------------------------
# Variants
#
# hc_available:  HC Valley Battery formula added (unused — shadow cost too high)
# ban_ya:        Yazhen A banned; hp absorbs freed cup; yc displaced by hp on ferr
# ban_yc:        Yazhen C banned; hp takes ferr+cup freed from yc; same $ as baseline
#                (hp+ya ≡ yc+ya for this resource mix)
# ban_hp:        Hetonite Parts banned; hp=0 in baseline → no change
# sc_cap_2:      SC Wuling Battery capped at floor(59/24)=2; lc picks up freed ori
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario,setup,expected_z,expected_mt,expected_rates,expected_slack,expected_dollar",
    [
        (
            "hc_available",
            {"add_hc": True},
            10,
            [0, 50, 0, 0],
            [59 / 24, 0, 0, 2, 2, 13 / 96, 14, 0, 0],
            [0, 0, 0, 0],
            2823 / 2,
        ),
        (
            "ban_ya",
            {"ya": 0},
            10,
            [0, 50, 0, 0],
            [59 / 24, 0, 13 / 24, 2, 0, 0, 14, 110],
            [0, 0, 0, 0],
            2801 / 2,
        ),
        (
            "ban_yc",
            {"yc": 0},
            10,
            [0, 50, 0, 0],
            [59 / 24, 0, 13 / 24, 2, 11 / 12, 0, 14, 0],
            [0, 0, 0, 0],
            2823 / 2,
        ),
        (
            "ban_hp",
            {"hp": 0},
            10,
            [0, 50, 0, 0],
            [59 / 24, 0, 0, 2, 2, 13 / 96, 14, 0],
            [0, 0, 0, 0],
            2823 / 2,
        ),
        (
            "sc_cap_2",
            {"sc": 2},
            10,
            [0, 50, 0, 0],
            [2, 11 / 18, 0, 2, 2, 1 / 4, 53 / 3, 0],
            [0, 0, 0, 0],
            4108 / 3,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_wuling_1p2d_variants(
    scenario,
    setup,
    expected_z,
    expected_mt,
    expected_rates,
    expected_slack,
    expected_dollar,
):
    f = _make_wuling_formulas()
    if setup.get("add_hc"):
        f["hc"] = make_formula([0, 180, 120, 0], output=54 * 6 * 1100 / 3200)
    for key in ("sc", "lc", "hp", "ya", "yc"):
        if key in setup:
            f[key].limit = setup[key]
    best, best_z, best_mt = _search(BASE_1P2D, f, max_forges=MAX_FORGES_1P2D)
    assert best.status == "optimal"
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)


# ---------------------------------------------------------------------------
# Effluent variants (5-resource model)
#
# Purification converts 24 xi → 30 eff.  SC needs 60 eff/run; HX needs 30 eff/run.
# Free initial effluent reduces Xi needed for purification.
#
# eff=0:   purify=83/12; xi_sell=14   (equivalent to 4-resource baseline, 2823/2)
# eff=30:  purify=71/12; xi_sell=38   (30 free eff saves 24 xi → xi_sell +24 → +24$/min)
# eff=60:  purify=59/12; xi_sell=62   (60 free eff saves 48 xi → xi_sell +48 → +48$/min)
# eff=90:  z drops 10→9; extra forge → hx=8/3; no xi_sell (all xi into purify+hx)
# eff=120: z=9; hx=62/21 (even more hx); dollar=21725/14
# ---------------------------------------------------------------------------

_BASE_1P2D_5R = np.array([0, 540, 90, 240, 0], dtype=float)
_XI5 = np.array([30, 0, 0, 0, 0], dtype=float)
_MTS5 = [[0, 50, 0, 0, 0], [0, 0, 25, 0, 0]]


def _make_5r_formulas():
    """5-resource formula set with explicit effluent tracking."""
    return {
        "purify": make_formula([24, 0, 0, 0, -30], output=0),
        "sc": make_formula([0, 240, 30, 0, 60], output=54 * 6),
        "lc": make_formula([30, 180, 0, 0, 0], output=25 * 6),
        "hp": make_formula([0, 0, 30, 240, 0], output=48 * 6),
        "hx": make_formula([60, 0, 0, 0, 30], output=27 * 6),
        "ya": make_formula([0, 0, 0, 120, 0], output=22 * 6),
        "yc": make_formula([0, 0, 120, 0, 0], output=16 * 6),
        "xi": make_formula([1, 0, 0, 0, 0], output=1),
        "cp": make_formula([0, 0, 0, 1, 0], output=1),
    }


def _search_5r(formulas, initial_eff=0, max_forges=MAX_FORGES_1P2D):
    base = _BASE_1P2D_5R.copy()
    base[4] = initial_eff
    candidates = []
    for z in range(max_forges + 1):
        formulas["hx"].limit = max_forges - z
        for mt in _MTS5:
            income = base + z * _XI5 + np.array(mt, dtype=float)
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, list(mt[:4])))
    return max(candidates, key=lambda r: r[0].dollar_output)


@pytest.mark.parametrize(
    "initial_eff,expected_z,expected_mt,expected_rates,expected_dollar",
    [
        # eff=0: purification must produce all needed eff from xi → equivalent to 4-resource
        (
            0,
            10,
            [0, 50, 0, 0],
            [83 / 12, 59 / 24, 0, 0, 2, 2, 13 / 96, 14, 0],
            2823 / 2,
        ),
        # eff=30: 30 free eff saves 24 xi from purification → xi_sell +24 → +24$/min
        (
            30,
            10,
            [0, 50, 0, 0],
            [71 / 12, 59 / 24, 0, 0, 2, 2, 13 / 96, 38, 0],
            2871 / 2,
        ),
        # eff=60: 60 free eff → xi_sell=62 (+48 from eff=0) → +48$/min
        (
            60,
            10,
            [0, 50, 0, 0],
            [59 / 12, 59 / 24, 0, 0, 2, 2, 13 / 96, 62, 0],
            2919 / 2,
        ),
        # eff=90: xi freed sufficiently to shift a forge from xi to hx; z=9, hx=8/3
        (
            90,
            9,
            [0, 50, 0, 0],
            [55 / 12, 59 / 24, 0, 0, 8 / 3, 2, 13 / 96, 0, 0],
            3011 / 2,
        ),
        # eff=120: z=9 still optimal; hx grows further to 62/21
        (
            120,
            9,
            [0, 50, 0, 0],
            [325 / 84, 59 / 24, 0, 0, 62 / 21, 2, 13 / 96, 0, 0],
            21725 / 14,
        ),
    ],
    ids=lambda x: f"eff{x}" if isinstance(x, int) else None,
)
def test_wuling_1p2d_effluent(
    initial_eff,
    expected_z,
    expected_mt,
    expected_rates,
    expected_dollar,
):
    f = _make_5r_formulas()
    best, best_z, best_mt = _search_5r(f, initial_eff=initial_eff)
    assert best.status == "optimal"
    assert best_z == expected_z
    assert np.allclose(best_mt, expected_mt)
    assert np.allclose(best.formula_rates, expected_rates)
    assert np.allclose(best.resource_slack, [0, 0, 0, 0, 0])
    assert np.isclose(best.dollar_output, expected_dollar)
