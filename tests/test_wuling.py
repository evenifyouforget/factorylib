import numpy as np
import pytest

from factorylib.optimize import Formula, maximize_dollar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_formula(consumption, output, limit=np.inf):
    return Formula(
        consumption=np.array(consumption, dtype=float), output=output, limit=limit
    )

# ---------------------------------------------------------------------------
# Wuling 1.2 - 8 Forge of the Sky, Heavy Xiranite, Hetonite Part
# ---------------------------------------------------------------------------

def test_wuling_1p2_full():
    # Global limit of 8 Forge of the Sky
    # If we use Z Forges for Xiranite, 8 - Z remain for Heavy Xiranite
    # 30 * Z Xiranite, 480 Originium Ore, 90 Ferrium Ore, 180 Cuprium Ore
    base_income = np.array([0, 480, 90, 180], dtype=float)
    # SC Wuling Battery
    # Worth $54, 6/min: 60 * 4/5 Xiranite, 240 Originium Ore, 30 Ferrium Ore
    sc = make_formula([60 * 4 / 5, 240, 30, 0], output=54 * 6)
    # LC Wuling Battery
    # Worth $25, 6/min: 30 Xiranite, 180 Originium Ore
    lc = make_formula([30, 180, 0, 0], output=25 * 6)
    # Hetonite Part
    # worth $48, 6/min: 30 Ferrium Ore, 240 Cuprium Ore
    hp = make_formula([0, 0, 30, 240], output=48 * 6)
    # Heavy Xiranite
    # Worth $27, 6/min: 60 + 60 * 4/5 Xiranite, Limit 8 - Z multiples
    hx = make_formula([60 + 60 * 4 / 5, 0, 0, 0], output=27 * 6)
    # Yazhen Syringe A
    # Worth $22, 6/min: 120 Cuprium Ore
    ya = make_formula([0, 0, 0, 120], output=22 * 6)
    # Yazhen Syringe C
    # Worth $16, 6/min: 120 Ferrium Ore
    yc = make_formula([0, 0, 120, 0], output=16 * 6)
    all_formulas = [sc, lc, hp, hx, ya, yc]
    candidates = []
    for z in range(9):
        hx.limit = 8 - z
        # Choice of metatransfer: 25 DOP (= 2 Originium Ore each), or 25 Ferrium Ore
        for metatransfer in [[0, 50, 0, 0], [0, 0, 25, 0]]:
            income = base_income + z * np.array([30, 0, 0, 0], dtype=float) + metatransfer
            result = maximize_dollar(income, all_formulas)
            candidates.append((result, z, metatransfer))
    best, best_z, best_mt = max(candidates, key=lambda r: r[0].dollar_output)
    assert best.status == "optimal"
    assert np.isclose(best.dollar_output, 1088.5)
    assert np.allclose(best.formula_rates, [53/24, 0, 0, 26/27, 3/2, 19/96])
    assert np.allclose(best.resource_slack, [0, 0, 0, 0])
    assert best_z == 7
    assert np.allclose(best_mt, [0, 50, 0, 0])
