"""Sanity tests for pp_goals_1p4.py -- the tier-shape generators
themselves (satisfaction_tiers/nonzero_production_tiers/
hard_satisfaction_bonus) are already covered in isolation by
test_pp_goals.py (reused unchanged, no dependency on the Wuling formula
set); these tests instead check pp_goals_1p4's own wiring against
wuling_1p4's real formula set.
"""

import numpy as np

from factorylib.endfield.pp_goals_1p4 import (
    ALL_NAMES,
    DOLLAR_EARNER_OUTPUTS,
    N,
    PPGoals1p4,
    build_pp_formulas,
    pp_supply,
)
from factorylib.endfield.wuling_1p4 import (
    RESOURCE_NAMES,
    WulingConfig1p4,
    build_formulas,
)
from factorylib.optimize import maximize_dollar


def test_all_names_extends_resource_names_with_flow_names_only():
    assert ALL_NAMES[: len(RESOURCE_NAMES)] == list(RESOURCE_NAMES)
    assert N == len(ALL_NAMES)


def test_dollar_earner_outputs_matches_real_formula_output():
    """Every DOLLAR_EARNER_OUTPUTS entry must match that formula's real
    $/multiple in wuling_1p4.build_formulas() -- if these drift apart,
    the CLI's dollar-vs-pp bookkeeping (see refine.py) silently
    misreports real $ income."""
    config = WulingConfig1p4()
    formulas = build_formulas(config)
    for name, dollar_per_multiple in DOLLAR_EARNER_OUTPUTS.items():
        assert name in formulas, f"{name!r} not in wuling_1p4.build_formulas()"
        assert formulas[name].output == dollar_per_multiple


def test_build_pp_formulas_is_feasible_and_solves():
    config = WulingConfig1p4()
    pp_goals = PPGoals1p4()
    formulas_dict = build_pp_formulas(config, pp_goals)
    formulas = list(formulas_dict.values())
    supply = pp_supply(config)
    result = maximize_dollar(supply, formulas)
    assert result.status == "optimal"
    assert result.dollar_output > 0.0


def test_new_1p4_nonzero_goals_are_present():
    """Every new_goals.md priority-list item (see module docstring) must
    materialize its own pp-tier formulas."""
    config = WulingConfig1p4()
    formulas = build_pp_formulas(config, PPGoals1p4())
    for prefix in (
        "pp_separator_core",
        "pp_cuprium_canister",
        "pp_cuprium_canister_inergen",
        "pp_cuprium_canister_xiragen",
        "pp_liquid_heavy_xiranite",
        "pp_pyrrolite_part",
        "pp_crafting_point",
        "pp_liquid_xiranite",
        "pp_pyrrolite",
    ):
        assert f"{prefix}_1" in formulas, f"missing tier formulas for {prefix!r}"


def test_component_item_tiers_key_directly_on_real_resource():
    """Resolves wuling_1p4's own open design question (see module
    docstring): Xiranite/Cuprium/Hetonite Component's Nonzero Production
    tiers must consume the real xiranite_component_item/etc. resource
    directly, not a redundant per-component flow dimension."""
    config = WulingConfig1p4()
    formulas = build_pp_formulas(config, PPGoals1p4())
    for prefix, resource_name in [
        ("pp_xiranite", "xiranite_component_item"),
        ("pp_cuprium", "cuprium_component_item"),
        ("pp_hetonite", "hetonite_component_item"),
        ("pp_pyrrolite", "pyrrolite_component"),
    ]:
        vec = formulas[f"{prefix}_1"].consumption
        assert vec[ALL_NAMES.index(resource_name)] > 0
        assert np.count_nonzero(vec) == 1


def test_hetonite_part_and_cuprium_component_first_caps_raised_from_1p2e_default():
    """new_goals.md: Hetonite Part and Cuprium Component are "Mid
    priority" (0.5/min) in 1.4, up from 1.2e's shared 0.1 default --
    Xiranite Component is NOT in that list and should stay at 0.1."""
    pp_goals = PPGoals1p4()
    assert pp_goals.hetonite_part_first_cap == 0.5
    assert pp_goals.cuprium_component_first_cap == 0.5
    assert pp_goals.xiranite_component_first_cap == 0.1
