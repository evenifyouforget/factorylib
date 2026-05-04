"""Shared helpers for Wuling 1.2 scenario tests."""
import numpy as np

from factorylib.optimize import Formula, maximize_dollar

BASE_INCOME = np.array([0, 480, 90, 180], dtype=float)
METATRANSFERS = [[0, 50, 0, 0], [0, 0, 25, 0]]


def make_formula(consumption, output, limit=np.inf):
    return Formula(
        consumption=np.array(consumption, dtype=float), output=output, limit=limit
    )


def _make_wuling_formulas(purification=True):
    """Return fresh dict of all 8 Wuling 1.2 formulas."""
    xi_sc = 60 * (4 / 5 if purification else 1)
    xi_hx = 60 + 30 * (4 / 5 if purification else 1)
    return {
        "sc": make_formula([xi_sc, 240, 30, 0], output=54 * 6),
        "lc": make_formula([30, 180, 0, 0], output=25 * 6),
        "hp": make_formula([0, 0, 30, 240], output=48 * 6),
        "hx": make_formula([xi_hx, 0, 0, 0], output=27 * 6),
        "ya": make_formula([0, 0, 0, 120], output=22 * 6),
        "yc": make_formula([0, 0, 120, 0], output=16 * 6),
        "xi": make_formula([1, 0, 0, 0], output=1),
        "cp": make_formula([0, 0, 0, 1], output=1),
    }


def _search(
    base_income, formulas, max_forges=8, metatransfers=None, fix_hx_limit=False
):
    """Search over forge allocations and metatransfer choices.

    Returns (best_result, best_z, best_mt).
    fix_hx_limit=True: do not override formulas["hx"].limit each iteration.
    """
    if metatransfers is None:
        metatransfers = METATRANSFERS
    candidates = []
    for z in range(max_forges + 1):
        if not fix_hx_limit:
            formulas["hx"].limit = max_forges - z
        for mt in metatransfers:
            income = base_income + z * np.array([30, 0, 0, 0], dtype=float) + mt
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, list(mt)))
    return max(candidates, key=lambda r: r[0].dollar_output)
