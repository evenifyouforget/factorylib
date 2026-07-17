"""Endfield 1.4 Wuling environment: extends the 1.2e model (wuling.py)
with 1.4's new recipes and materials, per tmp_notes/kaneko_1p4_data_sheet.md,
tmp_notes/1p4_new_features.md, tmp_notes/new_goals.md, and
tmp_notes/flexible_gear_crafting.md (repo-root-relative scratch notes,
not part of the installed package).

Every 1.2e formula/resource not explicitly changed by 1.4 is reused
unchanged (Cuprium/Ferrium refining, Sandleaf, SC/LC Wuling Battery, the
four Gear Components, the five power routes, etc.) -- this module only
replaces Forge of the Sky (now real recipes instead of an abstracted
allocation), unfolds Yazhen/Jincao into their real 3-stage recipes (see
below), and adds the new material chains on top.

Base supply (confirmed with the user): Originium Ore (540), Ferrium Ore
(120 -- wuling.py's own DEFAULT_BASE_SUPPLY uses 90 for 1.2e; that's
left untouched, 120 is specifically confirmed correct for 1.4), Cuprium
Ore (420), Inergen (>=260), and Xiragen (30) are the ONLY materials
with any base income at all, besides the virtual forge_budget
allowance. Every other new material must be crafted -- see
WulingConfig1p4's docstring for how Carbon/Stabilized Carbon's real
supply chain is now modeled, and Pyrrolite's remaining gap.

No folded formulas (confirmed with the user): 1.2e collapses Yazhen and
Jincao's Planting+Shredding+Reactor Crucible chain into one zero-cost
step each, since it's otherwise all-Water and there was no scarce
resource whose flow that would hide. That's no longer true once the
Carbon supply chain needs raw Yazhen/Jincao as an *alternative* use for
the same material -- so both are unfolded here into their real 3
stages (yazhen_plant/yazhen_powder_make/yazhen_solution_make and the
Jincao equivalents), verified to still reproduce 1.2e's historical
$-optimal figures exactly (see test_wuling_1p4.py).

Known gaps / assumptions (flag before treating any of this as final):
  - Pyrrolite has zero base supply (confirmed: it must be crafted, not
    sourced directly) and no confirmed recipe produces it either -- see
    WulingConfig1p4's docstring for the reverse-Transmuting-Unit-recipe
    gap this leaves. Structurally unreachable, along with everything
    downstream (Pyrrolite Part/Component, the T4 Crafting Point tier,
    pyrrolite_part_sell) until that's resolved.
  - Carbon/Stabilized Carbon: unlike Pyrrolite, this gap IS resolved --
    see WulingConfig1p4's docstring and the "no folded formulas" note
    above for the real 4-source supply chain (Buckflower, Sandleaf,
    Jincao, Yazhen).
  - The 6 "reverse" Fluid-Gas/Solid-Gas Transmuting Unit recipes ("every
    Z seconds", rate never given) are NOT modeled -- only the 11 forward
    recipes with concrete per-cycle numbers are. (This is also the only
    visible path to a Pyrrolite source, per the gap above.)
  - Filling Unit is only modeled for the 2 gas variants new_goals.md and
    the exploration-items list actually reference (Inergen, Xiragen),
    not all 8 possible gases the recipe is generically described for.
  - Every threshold/activation input (the "[threshold 6/min]" recipes:
    Fluid-Gas/Solid-Gas Transmuting Unit, Gas Reactor Globe) is folded in
    as a plain proportional consumption at the 6-per-30-throughput ratio,
    NOT the literal "fixed 6/min overhead regardless of building
    utilization" mechanic kaneko_1p4_data_sheet.md's science report
    describes. The literal mechanic is a per-building step function (a
    building either commits its full fixed activation cost or is idle),
    which would need the same two-layer integer-allocation pattern used
    for Forge of the Sky and Gas Dispersing Unit below -- skipped here
    for recipes with no stated building-count cap, since an unconstrained
    building count makes the two behave identically for LP-optimal
    purposes anyway (a rational plan never idles a partially-built
    activation for no reason) as long as nothing else caps how many
    buildings can exist. Revisit if a real building-count cap for any of
    these shows up later.
  - Forge of the Sky's Xiranite recipes use a two-layer capacity/run
    split (like 1.2e's existing Heavy Xiranite pattern) rather than
    folding real-material consumption directly into the integer
    allocation formula, specifically so a committed forge can still run
    below its max 30/min rate if Stabilized Carbon/Carbon supply is
    tight -- unlike 1.2e's old xiranite_forge_alloc, which had no real
    input to ever be short of.
  - Gas Dispersing Unit's own building count is unbounded here (no cap
    given, only the "13x13 fields can't overlap" spatial constraint,
    which -- like all physical topology/layout in this project -- isn't
    modeled). Each independent unit still yields exactly 4 environment
    allowance slots per the confirmed recipe ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factorylib.endfield import wuling as v1p2e
from factorylib.optimize import Formula, OptimizeResult, maximize_dollar

# New resource dimensions layered on top of wuling.RESOURCE_NAMES (1.2e's
# "xi"/"heavy_xiranite"/"hx_forge_capacity" dimensions are reused
# unchanged -- see module docstring for how Forge of the Sky's mechanic
# changes without renaming what it produces).
_NEW_RESOURCE_NAMES = [
    # Carbon/Stabilized Carbon supply chain (tmp_notes/old_prompt.md, the
    # original pre-1.2e problem statement -- 1.2e never needed this
    # chain since its Xiranite production was fully abstracted via
    # forge_budget, bypassing real Carbon inputs entirely). All 4 real
    # sources are modeled: Buckflower, Sandleaf (30:30), Jincao, Yazhen
    # (30:60, twice as efficient). The Jincao/Yazhen route requires
    # unfolding 1.2e's yazhen_solution_make/jincao_solution_make (which
    # collapse Planting+Shredding+Reactor Crucible into one zero-cost
    # step, since it's otherwise all-Water) back into their 3 real
    # stages -- confirmed with the user: no folded formulas should
    # remain, every original recipe gets modeled individually, verified
    # to still reproduce 1.2e's historical $-optimal figures exactly
    # (a lossless reformulation, same principle as origocrust_make/
    # ferrium_make's existing unconstrained 1:1 pass-throughs -- see
    # test_wuling_1p4.py's dedicated regression test for this).
    "buckflower",
    "yazhen_raw",
    "yazhen_powder",
    "jincao_raw",
    "jincao_powder",
    "carbon",
    "carbon_powder",
    "dense_carbon_powder",
    "stabilized_carbon",
    "pyrrolite",
    "inergen",
    "xiragen",
    # Gases produced in-model.
    "aquagen",
    "acridgen",
    "heavy_xiragen",
    "cuprium_gas",
    "hetonite_gas",
    "pyrrolite_gas",
    # New solids.
    "cuprium_canister",
    "cuprium_canister_inergen",
    "cuprium_canister_xiragen",
    "separator_core",
    "pyrrolite_part",
    "pyrrolite_component",
    "liquid_heavy_xiranite",
    # Gear-crafting chain (see flexible_gear_crafting.md). 1.2e's
    # xiranite_component/cuprium_component/hetonite_component formulas
    # have $0 output and no tracked resource of their own (their rate is
    # only ever scaled for display via wuling.GOOD_YIELD) -- these three
    # give them one, so component_to_tN below has something real to
    # consume (see build_formulas' override of those three formulas).
    "xiranite_component_item",
    "cuprium_component_item",
    "hetonite_component_item",
    "t1_crafting_point",
    "t2_crafting_point",
    "t3_crafting_point",
    "t4_crafting_point",
    # Forge of the Sky bookkeeping: two-layer capacity pools for the two
    # Xiranite recipes (hx_forge_capacity is reused from wuling.py
    # unchanged for the Heavy Xiranite recipe).
    "xi_forge_capacity",
    "xi_forge_stable_env_capacity",
    # Environment allowances (Gas Dispersing Unit -> gated recipes).
    "stable_env_allowance",
    "humid_env_allowance",
    "acrid_env_allowance",
    "xiranite_env_allowance",
    # Two-layer capacity pools for the other environment-gated recipes.
    "purification_hx_stable_capacity",
    "purification_hg_stable_capacity",
    "gas_reactor_globe_capacity",
]

RESOURCE_NAMES = v1p2e.RESOURCE_NAMES + _NEW_RESOURCE_NAMES

RESOURCE_LABELS = {
    **v1p2e.RESOURCE_LABELS,
    "buckflower": "Buckflower",
    "yazhen_raw": "Yazhen",
    "yazhen_powder": "Yazhen Powder",
    "jincao_raw": "Jincao",
    "jincao_powder": "Jincao Powder",
    "carbon": "Carbon",
    "carbon_powder": "Carbon Powder",
    "dense_carbon_powder": "Dense Carbon Powder",
    "stabilized_carbon": "Stabilized Carbon",
    "pyrrolite": "Pyrrolite",
    "inergen": "Inergen",
    "xiragen": "Xiragen",
    "aquagen": "Aquagen",
    "acridgen": "Acridgen",
    "heavy_xiragen": "Heavy Xiragen",
    "cuprium_gas": "Cuprium Gas",
    "hetonite_gas": "Hetonite Gas",
    "pyrrolite_gas": "Pyrrolite Gas",
    "cuprium_canister": "Cuprium Canister",
    "cuprium_canister_inergen": "Cuprium Canister filled with Inergen",
    "cuprium_canister_xiragen": "Cuprium Canister filled with Xiragen",
    "separator_core": "Separator Core",
    "pyrrolite_part": "Pyrrolite Part",
    "pyrrolite_component": "Pyrrolite Component",
    "liquid_heavy_xiranite": "Liquid Heavy Xiranite",
    "xiranite_component_item": "Xiranite Component",
    "cuprium_component_item": "Cuprium Component",
    "hetonite_component_item": "Hetonite Component",
    "t1_crafting_point": "T1 Crafting Point",
    "t2_crafting_point": "T2 Crafting Point",
    "t3_crafting_point": "T3 Crafting Point",
    "t4_crafting_point": "T4 Crafting Point",
    "xi_forge_capacity": "Forge of the Sky Xiranite Capacity",
    "xi_forge_stable_env_capacity": "Forge of the Sky Xiranite (Stable ENV) Capacity",
    "stable_env_allowance": "Stable Environment Allowance",
    "humid_env_allowance": "Humid Environment Allowance",
    "acrid_env_allowance": "Acrid Environment Allowance",
    "xiranite_env_allowance": "Xiranite Environment Allowance",
    "purification_hx_stable_capacity": (
        "Purification Unit Heavy Xiragen (Stable ENV) Capacity"
    ),
    "purification_hg_stable_capacity": (
        "Purification Unit Hetonite Gas (Stable ENV) Capacity"
    ),
    "gas_reactor_globe_capacity": "Gas Reactor Globe Capacity",
}

# Gases are transported by pipe, like liquids (belt_speed=120) -- not
# separately confirmed in-game, matching the same naming-convention
# inference wuling.py's own RESOURCE_BELT_SPEED docstring already makes
# for "Solution"/"Liquid X" items.
_GAS_NAMES = (
    "inergen",
    "xiragen",
    "aquagen",
    "acridgen",
    "heavy_xiragen",
    "cuprium_gas",
    "hetonite_gas",
    "pyrrolite_gas",
)

RESOURCE_BELT_SPEED = {
    **v1p2e.RESOURCE_BELT_SPEED,
    **{name: 120.0 for name in _GAS_NAMES},
    "buckflower": 30.0,
    "yazhen_raw": 30.0,
    "yazhen_powder": 30.0,
    "jincao_raw": 30.0,
    "jincao_powder": 30.0,
    "carbon": 30.0,
    "carbon_powder": 30.0,
    "dense_carbon_powder": 30.0,
    "stabilized_carbon": 30.0,
    "pyrrolite": 30.0,
    "cuprium_canister": 30.0,
    "cuprium_canister_inergen": 30.0,
    "cuprium_canister_xiragen": 30.0,
    "separator_core": 30.0,
    "pyrrolite_part": 30.0,
    "pyrrolite_component": 30.0,
    "liquid_heavy_xiranite": 120.0,
    # Crafting Points and every *_capacity/*_allowance dimension are pure
    # bookkeeping (no physical belt/pipe flow), like forge_budget/
    # hx_forge_capacity in wuling.py -- deliberately absent from this
    # dict, same convention.
}


def _extend(vec: np.ndarray) -> np.ndarray:
    """Pad a 1.2e-length consumption vector with zeros for the new 1.4
    resource dimensions."""
    out = np.zeros(len(RESOURCE_NAMES))
    out[: len(vec)] = vec
    return out


def make_formula(
    consumption: dict[str, float],
    output: float,
    limit: float = math.inf,
    integer: bool = False,
) -> Formula:
    """Same contract as wuling.make_formula, but keyed against this
    module's extended RESOURCE_NAMES."""
    vec = np.zeros(len(RESOURCE_NAMES), dtype=float)
    for name, amount in consumption.items():
        if name not in RESOURCE_NAMES:
            raise ValueError(f"Unknown resource name: {name!r}")
        vec[RESOURCE_NAMES.index(name)] = amount
    return Formula(consumption=vec, output=output, limit=limit, integer=integer)


# Confirmed "New Max Raw Material Income" figures (kaneko_1p4_data_sheet.md).
# Originium Ore/Ferrium Ore are listed as "unchanged" from 1.2e, but
# Ferrium Ore's confirmed number (120) does not match wuling.py's own
# DEFAULT_BASE_SUPPLY (90) -- confirmed with the user this is correct
# for 1.4 while 1.2e's own 90 stays untouched (not retroactively
# "fixed", since that would invalidate its own historical tests).
#
# Confirmed with the user: these 5 (Originium/Ferrium/Cuprium Ore,
# Inergen, Xiragen) are the ONLY materials with any base income at all
# (besides the virtual forge_budget allowance) -- Carbon, Stabilized
# Carbon, and Pyrrolite must all be crafted, never sourced directly.
# Carbon/Stabilized Carbon's real supply chain (tmp_notes/old_prompt.md,
# the original pre-1.2e problem statement) is now modeled below in
# build_formulas: Buckflower or Sandleaf -> Carbon -> Carbon Powder ->
# Dense Carbon Powder -> Stabilized Carbon. Pyrrolite still has no
# confirmed recipe producing it at all (see module docstring) and
# remains structurally unreachable, along with everything downstream
# (Pyrrolite Part, Pyrrolite Component, the T4 Crafting Point tier,
# pyrrolite_part_sell) -- not papered over with a guessed reverse-recipe
# ratio, since a wrong guess would propagate through its whole chain.
DEFAULT_ORIGINIUM_ORE = 540.0
DEFAULT_FERRIUM_ORE = 120.0
DEFAULT_CUPRIUM_ORE = 420.0
DEFAULT_INERGEN = 260.0  # "at least 260/min" per the data sheet
DEFAULT_XIRAGEN = 30.0
DEFAULT_MAX_FORGES = 12  # unchanged from 1.2e, confirmed with the user
# Planting Unit output for Buckflower/Yazhen/Jincao (see build_formulas'
# buckflower_plant/yazhen_plant/jincao_plant) has no real building-count
# cap given, same situation 1.2e's own sandleaf_plant is in -- reuses
# its exact rationale (see wuling.py's own comment on that formula), an
# arbitrary stand-in, not a real game constraint. See also
# tmp_notes/make_plants_not_free.md for a more principled alternative
# (an increasing pp cost per additional multiple) meant for whenever
# this gets a real pp_goals layer -- not implemented here since this
# module doesn't have one yet.
#
# 15, not 1.2e's own 5: 1.2e's sandleaf_plant=5 was "sized to comfortably
# cover its tracked consumers' floor demand" -- but that no longer holds
# once Sandleaf/Sandleaf Powder gets a brand-new competing consumer
# (the Carbon chain's dense_carbon_powder_make). Empirically, 10 is the
# exact minimum needed to still reproduce 1.2e's historical $-optimal
# figure exactly once Buckflower/Yazhen/Jincao/Sandleaf are all raised
# together (see test_wuling_1p4.py's regression test); 15 keeps a small
# margin rather than sitting right on that edge. build_formulas
# overrides 1.2e's inherited sandleaf_plant to this same value for
# consistency (it would otherwise stay stuck at 5).
DEFAULT_PLANTING_LIMIT = 15


def _default_base_supply() -> np.ndarray:
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ori")] = DEFAULT_ORIGINIUM_ORE
    supply[RESOURCE_NAMES.index("ferr")] = DEFAULT_FERRIUM_ORE
    supply[RESOURCE_NAMES.index("cup_ore")] = DEFAULT_CUPRIUM_ORE
    supply[RESOURCE_NAMES.index("inergen")] = DEFAULT_INERGEN
    supply[RESOURCE_NAMES.index("xiragen")] = DEFAULT_XIRAGEN
    return supply


@dataclass
class WulingConfig1p4:
    """Endfield 1.4 environment configuration.

    Args:
        base_supply: length-len(RESOURCE_NAMES) resource supply vector,
            in RESOURCE_NAMES order (see _default_base_supply for the
            confirmed-vs-placeholder figures).
        max_forges: number of Forge of the Sky units, split (as an
            integer MILP choice) between 3 recipes -- see build_formulas.
        metatransfers/purify_building/purify_node/secondary_goals/
            formula_limits/formula_outputs: same meaning as
            wuling.WulingConfig (passed straight through to the
            underlying 1.2e build_formulas() call this module extends).
            metatransfers defaults to wuling.DEFAULT_METATRANSFERS --
            nothing in the 1.4 source notes suggests this changed, and
            an earlier draft of this module set it to `[]` instead,
            which silently made metatransfer_option_0 -- used at rate
            1.0 in 1.2e's own historical $-optimal solution -- entirely
            unavailable. Caught by a regression test that reproduces
            that historical figure exactly.
    """

    base_supply: np.ndarray = field(default_factory=_default_base_supply)
    max_forges: int = DEFAULT_MAX_FORGES
    metatransfers: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array(mt, dtype=float) for mt in v1p2e.DEFAULT_METATRANSFERS
        ]
    )
    purify_building: bool = True
    purify_node: bool = True
    secondary_goals: bool = True
    formula_limits: dict[str, float] = field(default_factory=dict)
    formula_outputs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_supply = np.asarray(self.base_supply, dtype=float)
        if self.base_supply.shape != (len(RESOURCE_NAMES),):
            raise ValueError(
                f"base_supply must be length {len(RESOURCE_NAMES)}: "
                + ", ".join(RESOURCE_NAMES)
            )
        self.metatransfers = [np.asarray(mt, dtype=float) for mt in self.metatransfers]

    def _v1p2e_config(self) -> v1p2e.WulingConfig:
        return v1p2e.WulingConfig(
            base_supply=self.base_supply[: len(v1p2e.RESOURCE_NAMES)],
            max_forges=self.max_forges,
            metatransfers=self.metatransfers,
            purify_building=self.purify_building,
            purify_node=self.purify_node,
            secondary_goals=self.secondary_goals,
        )


def build_formulas(config: WulingConfig1p4) -> dict[str, Formula]:
    """Build the full 1.4 formula set: every 1.2e formula (extended to
    the new resource dimensions) except xiranite_forge_alloc/
    heavy_xiranite_forge_alloc (replaced below with real recipes), plus
    every new 1.4 formula. See module docstring for scope/assumptions.
    """
    base = v1p2e.build_formulas(config._v1p2e_config())
    f: dict[str, Formula] = {}
    # name -> (item resource, GOOD_YIELD) for the three component formulas
    # that need a real tracked output added (see _NEW_RESOURCE_NAMES'
    # comment on xiranite_component_item/etc.) -- ferrium_component has
    # no Crafting Point tier in the 1.4 scheme, so it's left alone.
    _component_item_yield = {
        "xiranite_component": (
            "xiranite_component_item",
            v1p2e.GOOD_YIELD["xiranite_component"],
        ),
        "cuprium_component": (
            "cuprium_component_item",
            v1p2e.GOOD_YIELD["cuprium_component"],
        ),
        "hetonite_component": (
            "hetonite_component_item",
            v1p2e.GOOD_YIELD["hetonite_component"],
        ),
    }
    # yazhen_solution_make/jincao_solution_make are replaced below with
    # their real 3-stage unfolding (Planting/Shredding/Reactor Crucible)
    # -- see _NEW_RESOURCE_NAMES' comment on why (no folded formulas).
    for name, formula in base.items():
        if name in (
            "xiranite_forge_alloc",
            "heavy_xiranite_forge_alloc",
            "yazhen_solution_make",
            "jincao_solution_make",
        ):
            continue
        vec = _extend(formula.consumption)
        if name in _component_item_yield:
            item_name, yield_per_multiple = _component_item_yield[name]
            vec[RESOURCE_NAMES.index(item_name)] = -yield_per_multiple
        # sandleaf_plant's inherited 1.2e limit (5) is raised to match
        # DEFAULT_PLANTING_LIMIT -- see that constant's own docstring
        # for why 1.4's new Carbon-chain demand on the same Sandleaf
        # pool means 5 is no longer enough to reproduce 1.2e's
        # historical $-optimal figure.
        limit = (
            float(DEFAULT_PLANTING_LIMIT) if name == "sandleaf_plant" else formula.limit
        )
        f[name] = Formula(
            consumption=vec,
            output=formula.output,
            limit=limit,
            integer=formula.integer,
        )

    # ---- Yazhen/Jincao unfolded into their real 3 stages (Planting,
    # Shredding, Reactor Crucible) instead of 1.2e's single folded
    # zero-cost step, so the Carbon chain below has real Yazhen/Jincao
    # to draw from. Ratios preserve 1.2e's exact "1 multiple = 30
    # Solution" convention (30 Yazhen Solution needs 30 Yazhen Powder
    # needs 15 Yazhen), so this is a lossless reformulation -- verified
    # against 1.2e's historical $-optimal figures in
    # test_wuling_1p4.py's dedicated regression test. Only the Planting
    # stage is capped (mirrors sandleaf_plant/buckflower_plant); the
    # unfolded Shredding/Reactor-Crucible stages stay uncapped like
    # every other lossless "_make" pass-through in this model.
    f["yazhen_plant"] = make_formula(
        {"yazhen_raw": -60.0}, 0.0, limit=DEFAULT_PLANTING_LIMIT
    )
    f["yazhen_powder_make"] = make_formula(
        {"yazhen_raw": 30.0, "yazhen_powder": -60.0}, 0.0
    )
    f["yazhen_solution_make"] = make_formula(
        {"yazhen_powder": 30.0, "yazhen_solution": -30.0}, 0.0
    )
    f["jincao_plant"] = make_formula(
        {"jincao_raw": -60.0}, 0.0, limit=DEFAULT_PLANTING_LIMIT
    )
    f["jincao_powder_make"] = make_formula(
        {"jincao_raw": 30.0, "jincao_powder": -60.0}, 0.0
    )
    f["jincao_solution_make"] = make_formula(
        {"jincao_powder": 30.0, "jincao_solution": -30.0}, 0.0
    )

    # ---- Carbon / Stabilized Carbon supply chain (tmp_notes/
    # old_prompt.md) -- 4 alternative raw sources (Buckflower, Sandleaf,
    # Jincao, Yazhen), all competing in the same $-maximizing LP with
    # whatever else those raw materials could otherwise be used for
    # (sandleaf_powder, the Syringe/Tea/Drink chains). Jincao/Yazhen are
    # twice as efficient (30:60) as Buckflower/Sandleaf (30:30). ----
    f["buckflower_plant"] = make_formula(
        {"buckflower": -30.0}, 0.0, limit=DEFAULT_PLANTING_LIMIT
    )
    f["carbon_from_buckflower"] = make_formula(
        {"buckflower": 30.0, "carbon": -30.0}, 0.0
    )
    f["carbon_from_sandleaf"] = make_formula(
        {"sandleaf_raw": 30.0, "carbon": -30.0}, 0.0
    )
    f["carbon_from_jincao"] = make_formula({"jincao_raw": 30.0, "carbon": -60.0}, 0.0)
    f["carbon_from_yazhen"] = make_formula({"yazhen_raw": 30.0, "carbon": -60.0}, 0.0)
    f["carbon_powder_make"] = make_formula(
        {"carbon": 30.0, "carbon_powder": -60.0}, 0.0
    )
    f["dense_carbon_powder_make"] = make_formula(
        {"carbon_powder": 60.0, "sandleaf": 30.0, "dense_carbon_powder": -30.0}, 0.0
    )
    f["stabilized_carbon_make"] = make_formula(
        {"dense_carbon_powder": 30.0, "stabilized_carbon": -30.0}, 0.0
    )

    # ---- Forge of the Sky: 12 buildings, 3 competing recipes ----
    # (see module docstring for the two-layer capacity/run rationale)
    f["xi_forge_alloc"] = make_formula(
        {"forge_budget": 1.0, "xi_forge_capacity": -30.0},
        0.0,
        limit=float(config.max_forges),
        integer=True,
    )
    f["xi_forge_run"] = make_formula(
        {"xi_forge_capacity": 1.0, "stabilized_carbon": 2.0, "xi": -1.0}, 0.0
    )
    f["xi_forge_stable_env_alloc"] = make_formula(
        {
            "forge_budget": 1.0,
            "stable_env_allowance": 1.0,
            "xi_forge_stable_env_capacity": -30.0,
        },
        0.0,
        limit=float(config.max_forges),
        integer=True,
    )
    f["xi_forge_stable_env_run"] = make_formula(
        {"xi_forge_stable_env_capacity": 1.0, "carbon": 1.0, "xi": -1.0}, 0.0
    )
    f["hx_forge_alloc"] = make_formula(
        {"forge_budget": 1.0, "hx_forge_capacity": -1.0},
        0.0,
        limit=float(config.max_forges),
        integer=True,
    )

    # ---- Gas Dispersing Unit: 4 environments, 6/min gas -> 4 allowance ----
    f["gas_dispersing_stable"] = make_formula(
        {"inergen": 6.0, "stable_env_allowance": -4.0}, 0.0, integer=True
    )
    f["gas_dispersing_humid"] = make_formula(
        {"aquagen": 6.0, "humid_env_allowance": -4.0}, 0.0, integer=True
    )
    f["gas_dispersing_acrid"] = make_formula(
        {"acridgen": 6.0, "acrid_env_allowance": -4.0}, 0.0, integer=True
    )
    f["gas_dispersing_xiranite_env"] = make_formula(
        {"xiragen": 6.0, "xiranite_env_allowance": -4.0}, 0.0, integer=True
    )

    # ---- New continuous recipes (no stated building-count cap) ----
    # Reactor Crucible: 1 Heavy Xiranite + 1 Acid -> 1 Liquid Heavy
    # Xiranite (Acid is free/unconstrained, matching wuling.py's own
    # Water/Acid convention -- see its module docstring). Mirrors 1.2e's
    # own liquid_xiranite_make (Xiranite -> Liquid Xiranite) exactly.
    # This was missing from an earlier draft, leaving
    # liquid_heavy_xiranite with no possible source at all and silently
    # making fluid_gas_heavy_xiragen permanently infeasible -- caught by
    # re-auditing the whole Xiranite/Heavy Xiranite solid/liquid/gas
    # family for the kind of "two different resources treated as
    # fungible" mistake this project has been bitten by before (the
    # historical DOP/Origocrust bug wuling.py's own module docstring
    # discusses); this is the opposite failure mode (a real distinction
    # accidentally left with no bridge at all, not two things wrongly
    # merged), but the same "audit every resource's actual production
    # path" fix applies.
    f["reactor_crucible_liquid_heavy_xiranite"] = make_formula(
        {"heavy_xiranite": 1.0, "liquid_heavy_xiranite": -1.0}, 0.0
    )
    f["fitting_unit"] = make_formula({"pyrrolite": 5.0, "pyrrolite_part": -1.0}, 0.0)
    f["moulding_unit"] = make_formula(
        {"cup": 2.0, "inergen": 1.0, "cuprium_canister": -1.0}, 0.0
    )
    f["gearing_unit"] = make_formula(
        {"pyrrolite": 1.0, "heavy_xiranite": 2.0, "pyrrolite_component": -1.0}, 0.0
    )
    f["filling_unit_inergen"] = make_formula(
        {
            "cuprium_canister": 1.0,
            "inergen": 1.0,
            "cuprium_canister_inergen": -1.0,
        },
        0.0,
    )
    f["filling_unit_xiragen"] = make_formula(
        {
            "cuprium_canister": 1.0,
            "xiragen": 1.0,
            "cuprium_canister_xiragen": -1.0,
        },
        0.0,
    )
    f["packaging_unit"] = make_formula(
        {"cuprium_canister": 1.0, "xi": 1.0, "separator_core": -2.0}, 0.0
    )

    # New Cloudseeder Station sellable goods (kaneko_1p4_data_sheet.md's
    # "Level 1: Sellable goods" list) -- SC Wuling Battery/Heavy
    # Xiranite/Yazhen Syringe A/Jincao Tea/Xiranite are already sellable
    # via 1.2e's own sc_sell/hx_sell/ya/jincao_tea/xi_sell formulas
    # (reused unchanged above); only Pyrrolite Part and Separator Core
    # are new. Both outposts modeled as one combined dollar pool (see
    # module docstring / confirmed with the user), so these compete for
    # the same $-maximizing objective as every other _sell formula.
    # 1 multiple = 1 item for both, since no batch-size convention is
    # given for them (unlike e.g. sc_sell's 6-item batches).
    f["pyrrolite_part_sell"] = make_formula({"pyrrolite_part": 1.0}, 70.0)
    f["separator_core_sell"] = make_formula({"separator_core": 1.0}, 1.0)
    f["purification_heavy_xiragen"] = make_formula(
        {"xiragen": 2.0, "separator_core": 2.0, "heavy_xiragen": -1.0}, 0.0
    )
    f["purification_hetonite_gas"] = make_formula(
        {"cuprium_gas": 2.0, "separator_core": 2.0, "hetonite_gas": -1.0}, 0.0
    )

    # Fluid-Gas Transmuting Unit (forward recipes only -- see module
    # docstring). Liquid Xiranite's [threshold 6/min] activation is
    # folded in proportionally at the 6-per-30-throughput ratio (1/5
    # unit per unit of real output), also per the module docstring.
    f["fluid_gas_aquagen"] = make_formula(
        {"liquid_xiranite": 0.2, "aquagen": -1.0}, 0.0
    )
    f["fluid_gas_xiragen"] = make_formula(
        {"liquid_xiranite": 1.2, "xiragen": -1.0}, 0.0
    )
    f["fluid_gas_cuprium_gas"] = make_formula(
        {
            "liquid_xiranite": 0.2,
            "cuprium_solution": 2.0,
            "cuprium_gas": -1.0,
        },
        0.0,
    )
    f["fluid_gas_acridgen"] = make_formula(
        {"liquid_xiranite": 0.2, "acridgen": -1.0}, 0.0
    )
    # Batch (per-cycle) ratios used directly here, not normalized to "per
    # 1 output unit" -- this recipe's 5-Heavy-Xiragen batch makes that
    # normalization error-prone (see module docstring's per-cycle
    # derivation): 1 cycle (every 10s) = 1 Liquid Xiranite (6/min
    # activation over 6 cycles/min) + 2 Liquid Heavy Xiranite -> 5 Heavy
    # Xiragen.
    f["fluid_gas_heavy_xiragen"] = make_formula(
        {
            "liquid_xiranite": 1.0,
            "liquid_heavy_xiranite": 2.0,
            "heavy_xiragen": -5.0,
        },
        0.0,
    )
    f["fluid_gas_hetonite_gas"] = make_formula(
        {
            "liquid_xiranite": 0.2,
            "hetonite_solution": 1.0,
            "hetonite_gas": -1.0,
        },
        0.0,
    )

    # Solid-Gas Transmuting Unit (forward recipes only). solid_gas_xiragen
    # is self-referential -- Xiragen is BOTH the activation input and the
    # recipe's own output -- so the net consumption entry combines
    # produced (-1.0) and activation-consumed (+0.2) into -0.8; forgetting
    # the production term entirely was a real bug caught while writing
    # this (see module docstring).
    f["solid_gas_xiragen"] = make_formula({"xi": 1.0, "xiragen": -0.8}, 0.0)
    # Batch (per-cycle) ratios, same reasoning as fluid_gas_heavy_xiragen
    # above: 1 cycle (every 10s) = 1 Xiragen (6/min activation over 6
    # cycles/min) + 2 Heavy Xiranite -> 5 Heavy Xiragen.
    f["solid_gas_heavy_xiragen"] = make_formula(
        {"xiragen": 1.0, "heavy_xiranite": 2.0, "heavy_xiragen": -5.0}, 0.0
    )
    f["solid_gas_cuprium_gas"] = make_formula(
        {"xiragen": 0.2, "cup": 2.0, "cuprium_gas": -1.0}, 0.0
    )
    f["solid_gas_hetonite_gas"] = make_formula(
        {"xiragen": 0.2, "hetonite": 1.0, "hetonite_gas": -2.0}, 0.0
    )
    f["solid_gas_pyrrolite_gas"] = make_formula(
        {"xiragen": 0.2, "pyrrolite": 1.0, "pyrrolite_gas": -1.0}, 0.0
    )

    # Gas Reactor Globe: Acrid ENV-gated, so it gets the two-layer
    # capacity/run pattern (unbounded building count, but the ENV
    # allowance backing it is itself limited).
    f["gas_reactor_globe_alloc"] = make_formula(
        {"acrid_env_allowance": 1.0, "gas_reactor_globe_capacity": -30.0},
        0.0,
        integer=True,
    )
    f["gas_reactor_globe_run"] = make_formula(
        {
            "gas_reactor_globe_capacity": 1.0,
            "hetonite_gas": 2.0,
            "xiragen": 1.0,
            "pyrrolite_gas": -1.0,
        },
        0.0,
    )

    # Stable-ENV Purification Unit variants (cheaper: 1 Separator Core
    # instead of 2, gated on Stable ENV instead of unconstrained).
    f["purification_heavy_xiragen_stable_alloc"] = make_formula(
        {
            "stable_env_allowance": 1.0,
            "purification_hx_stable_capacity": -30.0,
        },
        0.0,
        integer=True,
    )
    f["purification_heavy_xiragen_stable_run"] = make_formula(
        {
            "purification_hx_stable_capacity": 1.0,
            "xiragen": 2.0,
            "separator_core": 1.0,
            "heavy_xiragen": -1.0,
        },
        0.0,
    )
    f["purification_hetonite_gas_stable_alloc"] = make_formula(
        {
            "stable_env_allowance": 1.0,
            "purification_hg_stable_capacity": -30.0,
        },
        0.0,
        integer=True,
    )
    f["purification_hetonite_gas_stable_run"] = make_formula(
        {
            "purification_hg_stable_capacity": 1.0,
            "cuprium_gas": 2.0,
            "separator_core": 1.0,
            "hetonite_gas": -1.0,
        },
        0.0,
    )

    # ---- Gear-crafting Point chain (flexible_gear_crafting.md) ----
    f["component_to_t1"] = make_formula(
        {"xiranite_component_item": 50.0, "t1_crafting_point": -1.0}, 0.0
    )
    f["component_to_t2"] = make_formula(
        {"cuprium_component_item": 50.0, "t2_crafting_point": -1.0}, 0.0
    )
    f["component_to_t3"] = make_formula(
        {"hetonite_component_item": 50.0, "t3_crafting_point": -1.0}, 0.0
    )
    f["component_to_t4"] = make_formula(
        {"pyrrolite_component": 50.0, "t4_crafting_point": -1.0}, 0.0
    )
    f["t4_to_t3"] = make_formula(
        {"t4_crafting_point": 1.0, "t3_crafting_point": -2.0}, 0.0
    )
    f["t3_to_t2"] = make_formula(
        {"t3_crafting_point": 1.0, "t2_crafting_point": -5.0}, 0.0
    )
    f["t2_to_t1"] = make_formula(
        {"t2_crafting_point": 1.0, "t1_crafting_point": -1.0}, 0.0
    )

    for name, limit in config.formula_limits.items():
        if name not in f:
            raise ValueError(f"Unknown formula name in formula_limits: {name!r}")
        f[name].limit = limit
    for name, output in config.formula_outputs.items():
        if name not in f:
            raise ValueError(f"Unknown formula name in formula_outputs: {name!r}")
        f[name].output = output

    return f


def full_supply(config: WulingConfig1p4) -> np.ndarray:
    """config.base_supply plus the fixed amounts the forge-allocation and
    metatransfer-choice formulas compete over (max_forges of
    forge_budget, and -- if any metatransfer options exist -- exactly 1
    metatransfer_allowance) -- same role as wuling.full_supply. Missing
    the metatransfer_allowance credit here was a real bug: it silently
    made metatransfer_option_0 permanently unusable regardless of
    whether WulingConfig1p4.metatransfers was populated, caught by a
    regression test reproducing 1.2e's historical $-optimal figure."""
    supply = config.base_supply.copy()
    supply[RESOURCE_NAMES.index("forge_budget")] += config.max_forges
    if config.metatransfers:
        supply[RESOURCE_NAMES.index("metatransfer_allowance")] += 1.0
    return supply


def search(config: WulingConfig1p4) -> tuple[OptimizeResult, list[str]]:
    """Find the $-optimal 1.4 production plan. Returns (result,
    formula_names) since there's no metatransfer/z bookkeeping to
    surface specially (unlike wuling.SearchResult) -- every discrete
    choice here (forge allocation, environment allocation) is an
    ordinary named formula rate."""
    formulas = build_formulas(config)
    names = list(formulas.keys())
    supply = full_supply(config)
    result = maximize_dollar(supply, list(formulas.values()))
    return result, names
