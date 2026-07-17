"""Endfield 1.4 Wuling environment: extends the 1.2e model (wuling.py)
with 1.4's new recipes and materials, per tmp_notes/kaneko_1p4_data_sheet.md,
tmp_notes/1p4_new_features.md, tmp_notes/new_goals.md, and
tmp_notes/flexible_gear_crafting.md (repo-root-relative scratch notes,
not part of the installed package).

Every 1.2e formula/resource not explicitly changed by 1.4 is reused
unchanged (Cuprium/Ferrium refining, Sandleaf, Yazhen/Jincao, SC/LC
Wuling Battery, the four Gear Components, the five power routes, etc.)
-- this module only replaces Forge of the Sky (now real recipes instead
of an abstracted allocation) and adds the new material chains on top.

Known gaps / assumptions (flag before treating any of this as final):
  - Where raw Carbon, Stabilized Carbon, and Pyrrolite themselves come
    from is not given anywhere in the source notes -- modeled as new
    base-supply resources (like Originium/Ferrium/Cuprium Ore) with a
    large-but-finite (_PLACEHOLDER_SUPPLY) placeholder supply until real
    numbers are confirmed (NOT math.inf: Pyrrolite feeds a real $/item
    sell price, so an unconstrained supply of it makes the LP genuinely
    unbounded -- caught by search() failing to reach "optimal" status
    once that sell formula was added). See WulingConfig1p4's docstring.
  - The 6 "reverse" Fluid-Gas/Solid-Gas Transmuting Unit recipes ("every
    Z seconds", rate never given) are NOT modeled -- only the 11 forward
    recipes with concrete per-cycle numbers are.
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
  - Ferrium Ore's "New Max Raw Material Income" is listed as "120/min
    (unchanged)", but wuling.py's own DEFAULT_BASE_SUPPLY uses 90 --
    flagged, not silently resolved either way (1.2e's default is
    left alone; this module's own default uses the newly-confirmed 120).
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
    # New base/raw materials -- see module docstring's Carbon/Stabilized
    # Carbon/Pyrrolite gap.
    "carbon",
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
    "carbon": "Carbon",
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
    "carbon": 30.0,
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
# DEFAULT_BASE_SUPPLY (90) -- see module docstring, flagged rather than
# silently reconciled. Carbon/Stabilized Carbon/Pyrrolite have no source
# recipe or supply number given at all (see module docstring) --
# _PLACEHOLDER_SUPPLY stands in until real numbers are confirmed.
# Deliberately finite, not math.inf: Pyrrolite feeds pyrrolite_part_sell
# (a real $/item price), so an unconstrained supply of it would make the
# $-maximizing LP genuinely unbounded (caught by search() returning a
# non-optimal status once that sell formula was added) -- this is large
# enough to stay out of every other recipe's way for now, but is NOT a
# real number and must be replaced once real supply data exists.
DEFAULT_ORIGINIUM_ORE = 540.0
DEFAULT_FERRIUM_ORE = 120.0
DEFAULT_CUPRIUM_ORE = 420.0
DEFAULT_INERGEN = 260.0  # "at least 260/min" per the data sheet
DEFAULT_XIRAGEN = 30.0
DEFAULT_MAX_FORGES = 12  # unchanged from 1.2e, confirmed with the user
_PLACEHOLDER_SUPPLY = 10_000.0


def _default_base_supply() -> np.ndarray:
    supply = np.zeros(len(RESOURCE_NAMES))
    supply[RESOURCE_NAMES.index("ori")] = DEFAULT_ORIGINIUM_ORE
    supply[RESOURCE_NAMES.index("ferr")] = DEFAULT_FERRIUM_ORE
    supply[RESOURCE_NAMES.index("cup_ore")] = DEFAULT_CUPRIUM_ORE
    supply[RESOURCE_NAMES.index("inergen")] = DEFAULT_INERGEN
    supply[RESOURCE_NAMES.index("xiragen")] = DEFAULT_XIRAGEN
    supply[RESOURCE_NAMES.index("carbon")] = _PLACEHOLDER_SUPPLY
    supply[RESOURCE_NAMES.index("stabilized_carbon")] = _PLACEHOLDER_SUPPLY
    supply[RESOURCE_NAMES.index("pyrrolite")] = _PLACEHOLDER_SUPPLY
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
        purify_building/purify_node/secondary_goals/formula_limits/
            formula_outputs: same meaning as wuling.WulingConfig (passed
            straight through to the underlying 1.2e build_formulas()
            call this module extends). Metatransfer is not modeled --
            no 1.4-era equivalent has been given yet.
    """

    base_supply: np.ndarray = field(default_factory=_default_base_supply)
    max_forges: int = DEFAULT_MAX_FORGES
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

    def _v1p2e_config(self) -> v1p2e.WulingConfig:
        return v1p2e.WulingConfig(
            base_supply=self.base_supply[: len(v1p2e.RESOURCE_NAMES)],
            max_forges=self.max_forges,
            metatransfers=[],
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
    for name, formula in base.items():
        if name in ("xiranite_forge_alloc", "heavy_xiranite_forge_alloc"):
            continue
        vec = _extend(formula.consumption)
        if name in _component_item_yield:
            item_name, yield_per_multiple = _component_item_yield[name]
            vec[RESOURCE_NAMES.index(item_name)] = -yield_per_multiple
        f[name] = Formula(
            consumption=vec,
            output=formula.output,
            limit=formula.limit,
            integer=formula.integer,
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
    """config.base_supply plus the fixed amount the Forge of the Sky
    allocation formulas compete over (max_forges of forge_budget) --
    same role as wuling.full_supply."""
    supply = config.base_supply.copy()
    supply[RESOURCE_NAMES.index("forge_budget")] += config.max_forges
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
