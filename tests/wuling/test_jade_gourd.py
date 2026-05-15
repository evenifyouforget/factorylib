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
    ds = 0.0 if bc_only else 1.0  # dollar scale
    bc_eff = 1 if bc_only else bc  # cert value (1 cert = 1 unit in bc_only)
    f = {
        "sc": _f([xi_sc, 240, 30, 0, 0, 0], ds * 54 * 6),
        "lc": _f([30, 180, 0, 0, 0, 0], ds * 25 * 6),
        "hx_make": _f([xi_hx, 0, 0, 0, -6, 0], 0),
        "hx_sell": _f([0, 0, 0, 0, 6, 0], ds * 27 * 6),
        "ya": _f([0, 0, 0, 120, 0, 0], ds * 22 * 6),
        "yc": _f([0, 0, 120, 0, 0, 0], ds * 16 * 6),
        "xi_sell": _f([1, 0, 0, 0, 0, 0], ds),
        "cp_sell": _f([0, 0, 0, 1, 0, 0], ds),
        "hetonite_make": _f([0, 0, 30, 240, 0, -30], 0),
        "hp_sell": _f([0, 0, 0, 0, 0, 30], ds * 48 * 6),
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
            60,
            8,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 0, 0, 67 / 45, 0],
            [0, 0, 0, 0, 0, 0],
            7739 / 6,
        ),
        (
            61,
            7,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 1, 0, 11 / 10, 71 / 480, 0, 0, 1 / 5, 0, 0, 2 / 9, 1],
            [0, 0, 0, 0, 0, 0],
            38827 / 30,
        ),
        (
            111,
            7,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 1, 0, 11 / 10, 71 / 480, 0, 0, 1 / 5, 0, 0, 2 / 9, 1],
            [0, 0, 0, 0, 0, 0],
            47827 / 30,
        ),
        (
            112,
            6,
            [0, 0, 25, 0, 0, 0],
            [1 / 4, 0, 2, 0, 7 / 10, 0, 0, 0, 2 / 5, 0, 191 / 240, 0, 2],
            [0, 1107 / 4, 0, 0, 0, 0],
            1027863 / 640,
        ),
        # 120 = actual in-game price; above bp2 → same structure as 112
        (
            120,
            6,
            [0, 0, 25, 0, 0, 0],
            [1 / 4, 0, 2, 0, 7 / 10, 0, 0, 0, 2 / 5, 0, 191 / 240, 0, 2],
            [0, 1107 / 4, 0, 0, 0, 0],
            1089303 / 640,
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
            1,
            13,
            False,
            7,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 1, 0, 11 / 10, 71 / 480, 0, 0, 1 / 5, 0, 0, 2 / 9, 1],
            [0, 0, 0, 0, 0, 0],
            8287 / 6,
        ),
        # jg_bc=13, bc=6 > ~5.04 threshold → z=6 jg=2 (ferrium metatransfer)
        (
            6,
            13,
            False,
            6,
            [0, 0, 25, 0, 0, 0],
            [0, 0, 2, 0, 7 / 10, 0, 0, 0, 2 / 5, 0, 103 / 120, 2 / 15, 2],
            [0, 651 / 2, 0, 0, 0, 0],
            616703 / 320,
        ),
        # bc_only, jg_bc=12 < 38/3: xg (z=8) wins; cert output = 8/3*10*6 = 160
        (
            0,
            12,
            True,
            8,
            [0, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8 / 3, 0],
            [0, 530, 90, 180, 0, 0],
            160,
        ),
        # bc_only, jg_bc=13 > 38/3: jg (z=6) wins; cert output = 2/15*60 + 2*13*6 = 164
        # hetonite_make is degenerate: cuprium limits it to ≤3/4 but LP only needs ≥2/5.
        # expected_rates/slack are None; non-degenerate entries asserted separately.
        (
            0,
            13,
            True,
            6,
            [0, 50, 0, 0, 0, 0],
            None,
            None,
            164,
        ),
        # bc_only, jg_bc=30 (actual in-game bc yield): 30 >> 38/3 → jg (z=6) wins
        # cert output = 2/15*60 + 2*30*6 = 8 + 360 = 368
        # hetonite_make degenerate (2/5 ≤ hm ≤ 3/4); rates/slack None
        (
            0,
            30,
            True,
            6,
            [0, 50, 0, 0, 0, 0],
            None,
            None,
            368,
        ),
    ],
)
def test_jade_gourd_bc_worth(
    bc,
    jg_bc,
    bc_only,
    expected_z,
    expected_mt,
    expected_rates,
    expected_slack,
    expected_dollar,
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
        assert 2 / 5 <= hm <= 3 / 4 + 1e-9
    if expected_slack is not None:
        assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)


# ---------------------------------------------------------------------------
# Task 2: Alternate solutions from 120$ + 30bc baseline (bc=1)
#
# Baseline: z=6, ferrium MT, jg=2 (1319703/640 $/min)
# Fractional baseline rates: sc=1/4, ya=7/10, hetonite_make=2/5, hc=191/240.
# All floor to 0.
#
# Remove pipeline formulas (jg blocked entirely):
#   no_hetonite   – hetonite_make=hp_sell=0 → z=8 no-jg fallback (8275/6)
#
# Floor each fractional rate individually:
#   no_sc         – sc=0 → lc compensates partly (655359/320)
#   no_ya         – ya=0 → hetonite+hp_sell take up cup, but need extra ferr
#                   → hc shrinks; net loss 861/640 vs baseline (659421/320)
#   no_hc         – hc=0 → yc=191/240 absorbs freed ferr; same fraction! (10249/5)
#
# Combined floors:
#   floor_ya_hc   – ya=0 AND hc=0; same $ as no_hc (10249/5)
#                   When hc is absent, ya↔hetonite+hp_sell is exact: both use
#                   ferr+cup at same $/resource. With hc present, ferr is fully
#                   used → ya-to-hetonite swap requires shrinking hc → small loss.
#   floor_fracs   – sc=ya=hetonite_make=hc=0; jg blocked; lc/yc/cp_sell/xg (3616/3)
#
# Limit jg multiples:
#   jg_limit_1    – jg=1 run/min (6 jg/min); z=7, 55247/30
#   bc_cap_33     – jg=11/20 (3.3 jg/min, direct cap): binding, z=7 (238273/150)
#   bc_cap_6      – jg=1 (6 jg/min) = jg_limit_1 exactly
#
# bc sensitivity (same structure across all; only $ changes):
#   bc_low        – bc=0: cert value zeroed (1089303/640)
#   bc_high       – bc=5: amplified (2241303/640)
#   bc_only       – bc→∞; cert output=368
#
# Note: no_hxmake (custom loop) and bc-model shop saturation in separate tests.
# ---------------------------------------------------------------------------

_BASELINE_RATES = [1 / 4, 0, 2, 0, 7 / 10, 0, 0, 0, 2 / 5, 0, 191 / 240, 0, 2]
_BASELINE_SLACK = [0, 1107 / 4, 0, 0, 0, 0]
_NO_JG_RATES = [53 / 24, 0, 0, 0, 3 / 2, 19 / 96, 0, 0, 0, 0, 0, 67 / 45, 0]
_NO_JG_DOLLAR = 8275 / 6


@pytest.mark.parametrize(
    "scenario,bc,jg_bc,bc_only,mods,expected_z,expected_mt,expected_rates,expected_slack,expected_dollar",
    [
        (
            "baseline",
            1,
            30,
            False,
            {},
            6,
            [0, 0, 25, 0, 0, 0],
            _BASELINE_RATES,
            _BASELINE_SLACK,
            1319703 / 640,
        ),
        (
            "no_hetonite",
            1,
            30,
            False,
            {"hetonite_make": 0, "hp_sell": 0},
            8,
            [0, 50, 0, 0, 0, 0],
            _NO_JG_RATES,
            [0] * 6,
            _NO_JG_DOLLAR,
        ),
        # Floor each fractional baseline rate individually (all floor to 0).
        # sc=1/4, ya=7/10, hetonite_make=2/5, hc=191/240 all have floor=0.
        # no_sc: lc=2/5 picks up ori; hc→103/120 (more ferr freed from sc)
        (
            "no_sc",
            1,
            30,
            False,
            {"sc": 0},
            6,
            [0, 0, 25, 0, 0, 0],
            [0, 2 / 5, 2, 0, 7 / 10, 0, 0, 0, 2 / 5, 0, 103 / 120, 0, 2],
            [0, 507 / 2, 0, 0, 0, 0],
            655359 / 320,
        ),
        # no_ya: hetonite+hp_sell absorbs freed cup; BUT extra hetonite needs ferr
        # → hc shrinks by 7/80; net loss = 861/640 (ya is NOT equivalent to
        # hetonite here because ferr is fully used by hc)
        (
            "no_ya",
            1,
            30,
            False,
            {"ya": 0},
            6,
            [0, 0, 25, 0, 0, 0],
            [1 / 4, 0, 2, 0, 0, 0, 0, 0, 3 / 4, 7 / 20, 17 / 24, 0, 2],
            [0, 585 / 2, 0, 0, 0, 0],
            659421 / 320,
        ),
        # no_hc: freed ferr+ori → yc=191/240 (same fraction, just moved); same jg
        (
            "no_hc",
            1,
            30,
            False,
            {"hc": 0},
            6,
            [0, 0, 25, 0, 0, 0],
            [1 / 4, 0, 2, 0, 7 / 10, 191 / 240, 0, 0, 2 / 5, 0, 0, 0, 2],
            [0, 420, 0, 0, 0, 0],
            10249 / 5,
        ),
        # floor_ya_hc: floor ya AND hc simultaneously.
        # Without hc, ferr is free → ya↔hetonite+hp_sell holds exactly.
        # Same $ as no_hc: yc=17/24 absorbs same ferr that hc did.
        (
            "floor_ya_hc",
            1,
            30,
            False,
            {"ya": 0, "hc": 0},
            6,
            [0, 0, 25, 0, 0, 0],
            [1 / 4, 0, 2, 0, 0, 17 / 24, 0, 0, 3 / 4, 7 / 20, 0, 0, 2],
            [0, 420, 0, 0, 0, 0],
            10249 / 5,
        ),
        (
            "jg_limit_1",
            1,
            30,
            False,
            {"jg": 1},
            7,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 1, 0, 11 / 10, 71 / 480, 0, 0, 1 / 5, 0, 0, 2 / 9, 1],
            [0] * 6,
            55247 / 30,
        ),
        # floor all fractional baseline rates → sc=0, ya=0, hetonite_make=0, hc=0
        # jg blocked (no hetonite); lc+yc+cp_sell+xg regime takes over
        (
            "floor_fracs",
            1,
            30,
            False,
            {"sc": 0, "ya": 0, "hetonite_make": 0, "hc": 0},
            8,
            [0, 0, 25, 0, 0, 0],
            [0, 8 / 3, 0, 0, 0, 23 / 24, 0, 180, 0, 0, 0, 16 / 9, 0],
            [0] * 6,
            3616 / 3,
        ),
        # bc=0: jg_bc*bc=0, cert component vanishes; same structure as jg_price=120 bc=0
        (
            "bc_low",
            0,
            30,
            False,
            {},
            6,
            [0, 0, 25, 0, 0, 0],
            _BASELINE_RATES,
            _BASELINE_SLACK,
            1089303 / 640,
        ),
        (
            "bc_high",
            5,
            30,
            False,
            {},
            6,
            [0, 0, 25, 0, 0, 0],
            _BASELINE_RATES,
            _BASELINE_SLACK,
            2241303 / 640,
        ),
        # bc_cap_33: 3.3 jg/min = 11/20 run/min (6 items × 30 bc = 180 bc/run)
        # binding: baseline jg=2 runs/min exhausts shop in 3.5 days, not 13
        (
            "bc_cap_33",
            1,
            30,
            False,
            {"jg": 11 / 20},
            7,
            [0, 50, 0, 0, 0, 0],
            [
                53 / 24,
                0,
                11 / 20,
                0,
                32 / 25,
                409 / 2400,
                0,
                0,
                11 / 100,
                0,
                0,
                289 / 450,
                11 / 20,
            ],
            [0] * 6,
            238273 / 150,
        ),
        # bc_cap_6: 6 jg/min = 1 run/min; equals jg_limit_1 (same limit, same result)
        (
            "bc_cap_6",
            1,
            30,
            False,
            {"jg": 1},
            7,
            [0, 50, 0, 0, 0, 0],
            [53 / 24, 0, 1, 0, 11 / 10, 71 / 480, 0, 0, 1 / 5, 0, 0, 2 / 9, 1],
            [0] * 6,
            55247 / 30,
        ),
        # bc_only: same as bc_worth jg_bc=30; degenerate hetonite_make; cert output=368
        (
            "bc_only",
            0,
            30,
            True,
            {},
            6,
            [0, 50, 0, 0, 0, 0],
            None,
            None,
            368,
        ),
        # wuling_10pm: 10 jg/min = 5/3 runs/min, bc=0, no_hc, no_xg.
        # Cleanest solution in (6,12)/min range: all denominators ≤ 6.
        # z=6 ferr MT beats z=7 jg=1 by +29 $/min (crossover is between 8 and 9/min).
        # xi: 48*(5/6)+84*(5/3)=180; cup: 120*(5/6)+240*(1/3)=180;
        # ferr: 30*(5/6)+120*(2/3)+30*(1/3)=115. All binding.
        (
            "wuling_10pm",
            0,
            30,
            False,
            {"hc": 0, "xg": 0, "jg": 5 / 3},
            6,
            [0, 0, 25, 0, 0, 0],
            [5 / 6, 0, 5 / 3, 0, 5 / 6, 2 / 3, 0, 0, 1 / 3, 0, 0, 0, 5 / 3],
            [0, 280, 0, 0, 0, 0],
            1644,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_jade_gourd_120bc_alternates(
    scenario,
    bc,
    jg_bc,
    bc_only,
    mods,
    expected_z,
    expected_mt,
    expected_rates,
    expected_slack,
    expected_dollar,
):
    f = _make_formulas(jg_price=120, jg_bc=jg_bc, bc=bc, bc_only=bc_only)
    for key, val in mods.items():
        f[key].limit = val
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
        assert 2 / 5 <= hm <= 3 / 4 + 1e-9
    if expected_slack is not None:
        assert np.allclose(best.resource_slack, expected_slack)
    assert np.isclose(best.dollar_output, expected_dollar)


# ---------------------------------------------------------------------------
# no_hxmake: disable heavy xiranite production entirely.
# hx_make is always forced to 0; forges produce only extra xi (via XI_PER_FORGE).
# jg is impossible (needs heavy_xi); falls back to z=8 no-jg optimum.
# Requires a custom loop: _search overwrites hx_make.limit at each z.
# ---------------------------------------------------------------------------


def test_jade_gourd_120bc_no_hxmake():
    f = _make_formulas(jg_price=120, jg_bc=30, bc=1)
    candidates = []
    for z in range(9):
        f["hx_make"].limit = 0
        for mt in METATRANSFERS:
            income = BASE_INCOME + z * XI_PER_FORGE + np.array(mt, dtype=float)
            candidates.append((maximize_dollar(income, list(f.values())), z, list(mt)))
    best, best_z, best_mt = max(candidates, key=lambda r: r[0].dollar_output)
    assert best.status == "optimal"
    assert best_z == 8
    assert np.allclose(best_mt, [0, 50, 0, 0, 0, 0])
    assert np.allclose(best.formula_rates, _NO_JG_RATES)
    assert np.allclose(best.resource_slack, [0] * 6)
    assert np.isclose(best.dollar_output, _NO_JG_DOLLAR)


# ---------------------------------------------------------------------------
# BC intermediate resource model.
#
# Resources: [xi, ori, ferr, cup, heavy_xi, hetonite, bc_certs]
# bc_certs is an intermediate (supply=0):
#   jg produces 180 bc/run (6 items × 30 bc/item)
#   xg produces  60 bc/run (6 items × 10 bc/item)
#   bc_sell: consumes 1 bc → bc_sell_output $, capped at bc_sell_limit bc/min
#
# Separating cert production from cert valuation lets the LP choose how much
# of the bc budget to sell (vs waste) rather than capping jg/xg directly.
#
# Shop saturation: 1817000 bc / (13×24×60 min) ≈ 97 bc/min
# Baseline jg=2 → 360 bc/min >> 97 → shop exhausted in ~3.5 days.
#
# Key finding: with shop saturation cap, optimal PRODUCTION structure is unchanged
# (jg=2 still best for direct $). Only bc revenue is reduced.
# Compare to bc_cap_33 (direct jg cap): forces z=7, very different structure.
# ---------------------------------------------------------------------------

_BASE7 = np.append(BASE_INCOME, 0.0)
_XI7 = np.append(XI_PER_FORGE, 0.0)
_MTS7 = [mt + [0.0] for mt in METATRANSFERS]
_SHOP_BC_PER_MIN = 1817000 / (13 * 24 * 60)  # ≈ 97.06 bc/min


def _make_formulas_bc(jg_price=0, bc_sell_output=1, bc_sell_limit=np.inf):
    """Build 7-resource formula dict with bc_certs as intermediate (index 6).

    jg and xg produce bc as a byproduct; bc_sell converts bc → $ up to limit.
    """
    xi_sc = 60 * (4 / 5)
    xi_hx = 60 + 30 * (4 / 5)
    f = {
        "sc": _f([xi_sc, 240, 30, 0, 0, 0, 0], 54 * 6),
        "lc": _f([30, 180, 0, 0, 0, 0, 0], 25 * 6),
        "hx_make": _f([xi_hx, 0, 0, 0, -6, 0, 0], 0),
        "hx_sell": _f([0, 0, 0, 0, 6, 0, 0], 27 * 6),
        "ya": _f([0, 0, 0, 120, 0, 0, 0], 22 * 6),
        "yc": _f([0, 0, 120, 0, 0, 0, 0], 16 * 6),
        "xi_sell": _f([1, 0, 0, 0, 0, 0, 0], 1),
        "cp_sell": _f([0, 0, 0, 1, 0, 0, 0], 1),
        "hetonite_make": _f([0, 0, 30, 240, 0, -30, 0], 0),
        "hp_sell": _f([0, 0, 0, 0, 0, 30, 0], 48 * 6),
        "hc": _f([0, 180, 120, 0, 0, 0, 0], 54 * 6 * 1100 / 3200),
        "xg": _f([90, 0, 0, 0, 0, 0, -60], 40 * 6),
        "jg": _f([0, 0, 0, 0, 6, 6, -180], jg_price * 6),
        "bc_sell": _f([0, 0, 0, 0, 0, 0, 1], bc_sell_output, bc_sell_limit),
    }
    return f


def _search_bc(formulas, max_forges=8):
    candidates = []
    for z in range(max_forges + 1):
        formulas["hx_make"].limit = max_forges - z
        for mt in _MTS7:
            income = _BASE7 + z * _XI7 + np.array(mt, dtype=float)
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, list(mt[:6])))
    return max(candidates, key=lambda r: r[0].dollar_output)


@pytest.mark.parametrize(
    "bc_sell_limit,exp_bc_sell,exp_bc_slack,exp_dollar",
    [
        # uncapped: bc_sell sells all 360 bc/min; dollar = baseline
        (np.inf, 360, 0, 1319703 / 640),
        # shop saturation: only ~97 bc/min sold; 263/min wasted; jg=2 unchanged
        (_SHOP_BC_PER_MIN, _SHOP_BC_PER_MIN, 360 - _SHOP_BC_PER_MIN, 134716451 / 74880),
    ],
    ids=["uncapped", "shop_saturation"],
)
def test_jade_gourd_120bc_shop_saturation(
    bc_sell_limit, exp_bc_sell, exp_bc_slack, exp_dollar
):
    f = _make_formulas_bc(jg_price=120, bc_sell_output=1, bc_sell_limit=bc_sell_limit)
    best, best_z, best_mt = _search_bc(f)
    fkeys = list(f)
    assert best.status == "optimal"
    assert best_z == 6
    assert np.allclose(best_mt, [0, 0, 25, 0, 0, 0])
    # Production structure: indices 0-12 same as baseline (jg=2 optimal regardless)
    assert np.allclose(best.formula_rates[:13], _BASELINE_RATES)
    assert np.isclose(best.formula_rates[fkeys.index("bc_sell")], exp_bc_sell)
    assert np.allclose(best.resource_slack[:6], _BASELINE_SLACK)
    assert np.isclose(best.resource_slack[6], exp_bc_slack)
    assert np.isclose(best.dollar_output, exp_dollar)
