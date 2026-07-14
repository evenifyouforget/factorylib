"""Standard Wuling environment: configurable formula set + forge/metatransfer
search, generalizing the "1.2e full" model (tests/wuling/test_wuling_1p2e.py).

Resource vector (18): [xi, ori, ferr, cup_ore, cup, sew, eff, inert, dop,
                        sandleaf, origocrust, packed_origocrust, ferrium,
                        cuprium_part, xircon, hetonite, hetonite_part,
                        heavy_xiranite]
Formula order: cup_conv xi_sew ori_to_dop ferrium_make xircon_make sc lc
               hetonite_make hp_make hp_sell hx_make hx_sell ya yc xi_sell
               cuprium_part_make cp_sell [purify] [purify_node]
               [ferrium_component origocrust_make xiranite_component
               packed_origocrust_make cuprium_component hetonite_component
               sandleaf_powder thermal_bank]

Formulas are built with make_formula(), a name-keyed dict -> consumption
vector helper: it raises immediately on a typo'd resource name instead of
silently writing to the wrong array index (the exact bug class that
caused the earlier DOP/Origocrust fungibility bug, where two genuinely
different resources were folded into one index).

"dop" (Dense Originium Powder) is its own resource dimension, not folded
into "ori" (raw Originium Ore) equivalent cost as an earlier version of
this module did. SC/LC Wuling Battery actually consume DOP (30 Xircon +
120 DOP -> 6 SC; 30 Xiranite + 90 DOP -> 6 LC), while Ferrium/Xiranite
Component consume Origocrust/Packed Origocrust -- a *different* refining
chain off raw Ore that DOP cannot substitute for, and Thermal Bank burns
raw Ore directly. Folding DOP into a single fungible "ori" pool made a
metatransfer of DOP (see DEFAULT_METATRANSFERS) incorrectly spendable by
the Component/Thermal Bank formulas too. ori_to_dop (60 ori -> 30 dop,
lossless) lets locally-mined Ore still fund SC/LC exactly as before;
metatransferred dop bypasses that conversion, exactly as a real
metatransfer would. This changes nothing about $-optimal search()
results (ori_to_dop is a lossless pass-through with no other constraint,
so the net effect for local-only play is identical to the old direct
"ori" consumption) -- it only matters once Components exist to
(incorrectly, before this fix) compete for it.

"sandleaf" (Sandleaf Powder) is also its own resource dimension, not a
disconnected dead end. It's a real shared co-input to several Grinding
Unit recipes; ori_to_dop (Dense Originium Powder) and
packed_origocrust_make (Packed Origocrust) are the two tracked here, each
consuming 30 sandleaf per multiple. sandleaf_powder (the Planting Unit +
Shredding Unit formula) produces it (90 items/multiple, matching
GOOD_YIELD). Its other real consumers -- Dense Ferrium Powder for Steel,
Cryston Powder, Dense Carbon Powder, etc. -- still aren't modeled. Its
limit (5) is still just a modest building-count stand-in, not a real game
constraint, sized to comfortably cover its tracked consumers' floor
demand plus the delivery-job target, rather than letting the zero-cost
delivery-goods reward push it to an arbitrarily large excess (see
factorylib.endfield.goals's diminishing-but-never-penalized excess term).

Several materials that used to be folded into a base-resource-equivalent
cost are now their own resource dimensions, produced by a dedicated
"_make" formula and consumed by whatever needs them, instead of being
approximated: Origocrust/Packed Origocrust (Ferrium/Xiranite Component's
real inputs), Ferrium (refined, distinct from Ferrium Ore -- needed by
both Ferrium Component and the Xircon reaction), Cuprium Part (distinct
from Cuprium -- needed by Cuprium Component and cp_sell), Xircon
(previously folded into sc), and Hetonite/Hetonite Part/Heavy Xiranite
(previously folded into hp/hx). Each "_make" formula is a lossless,
unconstrained pass-through (limit=inf, no other constraint), so none of
this changes any $-optimal search() result -- every historical
scenario-equivalence figure (206735/146, 2823/2, 2229/2, etc.) is
unchanged; it only makes the intermediate resource visible to the
fitness/CLI/delivery-job layers instead of hidden inside a collapsed
coefficient. sc keeps its own name (30 xircon + 120 dop -> 6 SC); hp/hx
split into hetonite_make + hp_make + hp_sell and hx_make + hx_sell
respectively (naming matches the split already used in
tests/wuling/test_jade_gourd.py), each now $-bearing only at the
"_sell" step -- WulingConfig.fix_hx_limit and SELL_PRIORITY reference
hx_make/hp_sell/hx_sell accordingly, not the old bare "hx"/"hp".

The bracketed "secondary goals" formulas (gated by
WulingConfig.secondary_goals, on by default) exist purely to give the
Part 4/5 fitness function's gear/delivery/power terms something to act
on -- they all have $ output=0, so the raw dollar-maximizing LP in
search() never chooses to run them (any positive rate would only spend
resources the $-formulas are already fully using, at zero marginal $
value), and none of the existing scenario-equivalence tests change:
  - hetonite_component / cuprium_component / xiranite_component /
    ferrium_component now consume the real de-flattened intermediates
    directly (e.g. hetonite_component is an exact 1:1 match to the recipe
    line: 12 Hetonite Part + 12 Heavy Xiranite -> 6 Hetonite Component,
    corrected from the raw recipe list's apparent typo "-> 6 Hetonite
    Part"). origocrust_make and packed_origocrust_make are gated
    alongside them since nothing else consumes Origocrust/Packed
    Origocrust; Ferrium/Cuprium Part are core (see above) since sc/hp_sell
    /cp_sell also need them. Cryston Component and Amethyst Component are
    still NOT modeled: their chains need Amethyst Ore, a base resource
    this model doesn't track at all.
  - sandleaf_powder: Planting Unit + Shredding Unit collapsed into one
    formula that consumes none of the *tracked base* resources (matching
    the spec's "very cheap material" framing) but does produce the
    "sandleaf" resource dimension several other formulas need -- see
    above. Its limit represents a modest, arbitrary number of building
    instances (this LP has no building-count dimension), not a real game
    constraint.
  - thermal_bank: the simplest Thermal Bank recipe (raw Originium Ore ->
    W), tracked via POWER_YIELD below since Formula.output is $-only.
    The more resource-efficient battery -> power route (spec: "1.5 SC
    Wuling Battery -> 3200 W") is NOT modeled -- it would need its own
    _power formula per battery type; left as a future extension.

Being zero-$, these formulas are also zero-$ *ties* with doing nothing,
above whatever floor the $-maximizing LP actually needs from them.
sandleaf_powder is the one exception with a genuine floor now: ori_to_dop
(and, when xiranite_component is active, packed_origocrust_make) needs 30
sandleaf per multiple, so at their LP-chosen rates, sandleaf_powder's rate
is pinned to at least that floor -- but any excess above it remains a
real LP degeneracy (any value up to its limit is equally optimal at $0
marginal value), same as ferrium_component / xiranite_component /
cuprium_component / hetonite_component / thermal_bank always are. Not an
economically meaningful "tied solution" in the sense Part 2 was designed
for (a genuine choice between two strategies).
factorylib.endfield.cli filters SECONDARY_GOAL_FORMULA_NAMES out of its
tied-alternatives search for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factorylib.optimize import Formula, OptimizeResult, maximize_dollar

# Formulas that exist only to give the Part 4/5 fitness function
# something to act on (see module docstring); all have $ output=0.
SECONDARY_GOAL_FORMULA_NAMES = (
    "ferrium_component",
    "xiranite_component",
    "cuprium_component",
    "hetonite_component",
    "sandleaf_powder",
    "thermal_bank",
)

RESOURCE_NAMES = [
    "xi",
    "ori",
    "ferr",
    "cup_ore",
    "cup",
    "sew",
    "eff",
    "inert",
    "dop",
    "sandleaf",
    "origocrust",
    "packed_origocrust",
    "ferrium",
    "cuprium_part",
    "xircon",
    "hetonite",
    "hetonite_part",
    "heavy_xiranite",
]
FORMULA_NAMES = [
    "cup_conv",
    "xi_sew",
    "ori_to_dop",
    "ferrium_make",
    "xircon_make",
    "sc",
    "lc",
    "hetonite_make",
    "hp_make",
    "hp_sell",
    "hx_make",
    "hx_sell",
    "ya",
    "yc",
    "xi_sell",
    "cuprium_part_make",
    "cp_sell",
    "purify",
    "purify_node",
    "ferrium_component",
    "origocrust_make",
    "xiranite_component",
    "packed_origocrust_make",
    "cuprium_component",
    "hetonite_component",
    "sandleaf_powder",
    "thermal_bank",
]

# Belts (solids) run at 30 items/min; pipes (liquids) run at 120 items/min
# (see factorylib_tmp_physical_factory_construction.md). Used to price a
# resource flow's physical complexity in terms of "how many belts/pipes"
# it represents, rather than the abstract recipe-multiple fraction alone
# -- see factorylib.endfield.goals.fitness.
RESOURCE_BELT_SPEED = {
    "xi": 30.0,
    "ori": 30.0,
    "ferr": 30.0,
    "cup_ore": 30.0,
    "cup": 30.0,
    "sew": 120.0,
    "eff": 120.0,
    "inert": 120.0,
    "dop": 30.0,
    "sandleaf": 30.0,
    "origocrust": 30.0,
    "packed_origocrust": 30.0,
    "ferrium": 30.0,
    "cuprium_part": 30.0,
    "xircon": 30.0,
    "hetonite": 30.0,
    "hetonite_part": 30.0,
    "heavy_xiranite": 30.0,
}

# W produced per multiple of a formula's rate. Formula.output is $-only, so
# a formula that only contributes power (no $ value) is tracked here
# instead -- see plan_from_search_result in factorylib.endfield.goals.
POWER_YIELD = {"thermal_bank": 50.0}

# Items of the named good produced per multiple of a formula's rate (the
# "6" or "90" etc. in "-> 6 Ferrium Component" / "-> 90 Sandleaf Powder").
# WulingGoals.delivery_goods/gear_min_target are expressed in real items/
# min (matching the spec's own units, e.g. "0.5/min of Cuprium
# Component"), not raw recipe multiples -- plan_from_search_result and
# refine._plan_from_rates scale by this before comparing. Also covers the
# $-earning formulas (their "6" or "1" is otherwise only implicit in
# Formula.output, e.g. sc's output=54*6 -- not separable back into a
# per-item price and a batch size without this), so the CLI can show a
# real items/min figure next to a formula's "multiples" count instead of
# requiring the batch size to be known/assumed by the reader.
GOOD_YIELD = {
    "sandleaf_powder": 90.0,
    "ferrium_component": 6.0,
    "xiranite_component": 6.0,
    "cuprium_component": 6.0,
    "hetonite_component": 6.0,
    "sc": 6.0,
    "lc": 6.0,
    "hp_sell": 6.0,
    "hx_sell": 6.0,
    "ya": 6.0,
    "yc": 6.0,
    "xi_sell": 1.0,
    "cp_sell": 1.0,
}

# The outpost's $ savings only regenerate at a fixed rate (WulingGoals.
# stock_bill_cap), so at any instant only that much $ worth of produced
# goods can actually be sold -- the rest accumulates unsold rather than
# being sold at a discount. Real play sells in this order because
# letting Hetonite Part or Yazhen A pile up to their storage cap stalls
# those production lines entirely, and that stall cascades into less
# Sewage/Xircon Effluent for the battery recipes that depend on it;
# Heavy Xiranite and SC Wuling Battery are next (still worth protecting,
# but less immediately disruptive), then Yazhen C. Everything else (LC
# Wuling Battery, Xiranite/Cuprium Part sold) has no such upstream
# dependency, so it's lowest priority by default -- see
# factorylib.priority_sell.allocate_by_priority.
SELL_PRIORITY = ("ya", "hp_sell", "hx_sell", "sc", "yc")

RESOURCE_LABELS = {
    "xi": "Xiranite",
    "ori": "Originium Ore",
    "ferr": "Ferrium Ore",
    "cup_ore": "Cuprium Ore",
    "cup": "Cuprium",
    "sew": "Sewage",
    "eff": "Xircon Effluent",
    "inert": "Inert Xircon Effluent",
    "dop": "Dense Originium Powder",
    "sandleaf": "Sandleaf Powder",
    "origocrust": "Origocrust",
    "packed_origocrust": "Packed Origocrust",
    "ferrium": "Ferrium",
    "cuprium_part": "Cuprium Part",
    "xircon": "Xircon",
    "hetonite": "Hetonite",
    "hetonite_part": "Hetonite Part",
    "heavy_xiranite": "Heavy Xiranite",
}

FORMULA_LABELS = {
    "cup_conv": "Cuprium Ore Refining (Cuprium Ore → Cuprium + Sewage)",
    "xi_sew": "Xiranite + Sewage Reaction (→ Xircon Effluent + Inert Xircon Effluent)",
    "ori_to_dop": "Originium Ore Grinding (→ Dense Originium Powder)",
    "ferrium_make": "Ferrium Refining (Ferrium Ore → Ferrium)",
    "xircon_make": "Xircon Reaction (Xircon Effluent + Ferrium → Xircon + Sewage)",
    "sc": "SC Wuling Battery",
    "lc": "LC Wuling Battery",
    "hetonite_make": "Hetonite Refining (Cuprium + Ferrium → Hetonite + Sewage)",
    "hp_make": "Hetonite Part Assembly (Hetonite → Hetonite Part)",
    "hp_sell": "Hetonite Part (sold)",
    "hx_make": "Heavy Xiranite Assembly (Xiranite + Xircon Effluent → Heavy Xiranite)",
    "hx_sell": "Heavy Xiranite (sold)",
    "ya": "Yazhen Syringe A",
    "yc": "Yazhen Syringe C",
    "xi_sell": "Xiranite (sold)",
    "cuprium_part_make": "Cuprium Part Fitting (Cuprium → Cuprium Part)",
    "cp_sell": "Cuprium Part (sold)",
    "purify": "Purification Building (Inert Xircon Effluent → Xircon Effluent)",
    "purify_node": "Test Area Purification Node (Sewage → Xircon Effluent)",
    "ferrium_component": "Ferrium Component",
    "origocrust_make": "Origocrust Refining (Originium Ore → Origocrust)",
    "xiranite_component": "Xiranite Component",
    "packed_origocrust_make": (
        "Packed Origocrust Dilution (Origocrust + Sandleaf Powder → Packed Origocrust)"
    ),
    "cuprium_component": "Cuprium Component",
    "hetonite_component": "Hetonite Component",
    "sandleaf_powder": "Sandleaf Powder",
    "thermal_bank": "Thermal Bank (Originium Ore → Power)",
}

# Metatransfer choices are literal items selected in the game's
# Metatransfer menu, at their own real quantity -- NOT the internal
# resource-equivalent units used elsewhere in this module. Maps each
# resource dimension that DEFAULT_METATRANSFERS actually uses to
# (item name, real item quantity per 1 unit of that resource dimension),
# so the CLI can report what to actually select rather than a raw vector.
METATRANSFER_ITEMS = {
    "dop": "Dense Originium Powder",
    "ferr": "Ferrium Ore",
}

DEFAULT_BASE_SUPPLY = (0, 540, 90, 240) + (0,) * (len(RESOURCE_NAMES) - 4)  # 1.2e base
DEFAULT_MAX_FORGES = 12
DEFAULT_METATRANSFERS = (
    # 25 Dense Originium Powder
    (0, 0, 0, 0, 0, 0, 0, 0, 25) + (0,) * (len(RESOURCE_NAMES) - 9),
    # 25 Ferrium Ore
    (0, 0, 25) + (0,) * (len(RESOURCE_NAMES) - 3),
)
XI_PER_FORGE = np.array([30.0] + [0.0] * (len(RESOURCE_NAMES) - 1))


@dataclass
class WulingConfig:
    """Standard Wuling environment configuration.

    Defaults reproduce "1.2e full" (everything on) exactly.

    Args:
        base_supply: length-len(RESOURCE_NAMES) resource supply vector
            (before forge/metatransfer top-ups), in RESOURCE_NAMES order.
        max_forges: number of Forge of the Sky units to split between
            Xiranite supply (z) and Heavy Xiranite capacity (max_forges - z).
        metatransfers: alternate resource top-up choices searched alongside z.
        purify_building: whether the Purification Building formula (inert ->
            eff) is included.
        purify_node: whether the Test Area Purification Node formula
            (sew -> eff, max 12 multiples) is included.
        secondary_goals: whether the Part 4/5 gear/delivery/power formulas
            (Components, Sandleaf Powder, Thermal Bank -- see module
            docstring) are included. They never affect $-optimal search()
            results (zero $ output), only what the Part 4/5 fitness
            function and refine() can act on.
        formula_limits: per-formula `limit` overrides, e.g. {"ya": 0} to ban
            a formula. Keys must be in FORMULA_NAMES.
        formula_outputs: per-formula `output` ($/run) overrides. Keys must be
            in FORMULA_NAMES.
        fix_hx_limit: if True, search() will not override
            formulas["hx_make"].limit each z iteration (use the
            configured/overridden limit as-is).
    """

    base_supply: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_BASE_SUPPLY, dtype=float)
    )
    max_forges: int = DEFAULT_MAX_FORGES
    metatransfers: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array(mt, dtype=float) for mt in DEFAULT_METATRANSFERS
        ]
    )
    purify_building: bool = True
    purify_node: bool = True
    secondary_goals: bool = True
    formula_limits: dict[str, float] = field(default_factory=dict)
    formula_outputs: dict[str, float] = field(default_factory=dict)
    fix_hx_limit: bool = False

    def __post_init__(self) -> None:
        self.base_supply = np.asarray(self.base_supply, dtype=float)
        if self.base_supply.shape != (len(RESOURCE_NAMES),):
            raise ValueError(
                f"base_supply must be length {len(RESOURCE_NAMES)}: "
                + ", ".join(RESOURCE_NAMES)
            )
        self.metatransfers = [np.asarray(mt, dtype=float) for mt in self.metatransfers]
        for name in self.formula_limits:
            if name not in FORMULA_NAMES:
                raise ValueError(f"Unknown formula name in formula_limits: {name!r}")
        for name in self.formula_outputs:
            if name not in FORMULA_NAMES:
                raise ValueError(f"Unknown formula name in formula_outputs: {name!r}")


def make_formula(
    consumption: dict[str, float], output: float, limit: float = np.inf
) -> Formula:
    """Build a Formula from a name-keyed consumption dict instead of a
    positional array -- raises immediately on an unknown resource name
    rather than silently writing to the wrong index (see module
    docstring). Positive = consumed, negative = produced, matching
    Formula.consumption's convention."""
    vec = np.zeros(len(RESOURCE_NAMES), dtype=float)
    for name, amount in consumption.items():
        if name not in RESOURCE_NAMES:
            raise ValueError(f"Unknown resource name: {name!r}")
        vec[RESOURCE_NAMES.index(name)] = amount
    return Formula(consumption=vec, output=output, limit=limit)


def build_formulas(config: WulingConfig) -> dict[str, Formula]:
    """Build a fresh dict of 1.2e-model Formulas per config.

    Ports tests/wuling/test_wuling_1p2e.py::_make_1p2e_formulas (with DOP
    and the de-flattened intermediates split into their own resource
    dimensions -- see module docstring), then applies
    config.formula_limits / config.formula_outputs overrides.
    """
    # ori_to_dop's Sandleaf Powder co-input is only tracked when
    # secondary_goals is on -- that's the only mode where anything
    # produces "sandleaf" at all (base_supply never does; it's not a
    # mined resource). With secondary_goals off, this reverts to the
    # original collapsed convention of not tracking that co-input, so
    # disabling secondary_goals still never changes $-optimal search()
    # results (see test_secondary_goals_never_change_1p2e_full_dollar_output).
    _ori_to_dop_sandleaf = 30.0 if config.secondary_goals else 0.0
    f = {
        # 30 cup_ore -> 30 cup + 30 sew
        "cup_conv": make_formula({"cup_ore": 30, "cup": -30, "sew": -30}, 0),
        # 30 xi + 30 sew -> 30 eff + 30 inert
        "xi_sew": make_formula({"xi": 30, "sew": 30, "eff": -30, "inert": -30}, 0),
        # 60 ori (+ 30 sandleaf, if secondary_goals) -> 30 dop (collapses 2
        # Shredding Units, each 30 ori -> 30 Originium Powder, feeding 1
        # Grinding Unit, 60 Powder + 30 Sandleaf Powder -> 30 DOP -- see
        # module docstring). Scaled to belt size (30/min) like every other
        # formula's "1 multiple" -- the minimally-reduced 2:1 ratio alone
        # ("2 ori -> 1 dop") made a formula's rate mean "2 ore/min" here
        # instead of a recognizable physical unit everywhere else.
        "ori_to_dop": make_formula(
            {"ori": 60, "dop": -30, "sandleaf": _ori_to_dop_sandleaf}, 0
        ),
        # Ferrium Refining: 30 Ferrium Ore -> 30 Ferrium. Core (not
        # secondary-gated): both the Xircon reaction below and
        # ferrium_component need it.
        "ferrium_make": make_formula({"ferr": 30, "ferrium": -30}, 0),
        # Xircon Reaction: 60 eff + 30 Ferrium -> 30 Xircon + 30 sew
        # (previously folded directly into "sc").
        "xircon_make": make_formula(
            {"eff": 60, "ferrium": 30, "xircon": -30, "sew": -30}, 0
        ),
        # SC: 30 Xircon + 120 dop -> 6 SC
        "sc": make_formula({"xircon": 30, "dop": 120}, 54 * 6),
        # LC: 30 xi + 90 dop -> 6 LC
        "lc": make_formula({"xi": 30, "dop": 90}, 25 * 6),
        # Hetonite Refining: 240 cup + 30 Ferrium -> 30 Hetonite + 30 sew
        # (previously folded directly into "hp"). Core: hp_sell needs it.
        "hetonite_make": make_formula(
            {"cup": 240, "ferrium": 30, "hetonite": -30, "sew": -30}, 0
        ),
        # Hetonite Part Assembly: 30 Hetonite -> 6 Hetonite Part.
        "hp_make": make_formula({"hetonite": 30, "hetonite_part": -6}, 0),
        # Sell 6 Hetonite Part at $48/unit ($288/multiple, matching hp's
        # original combined output).
        "hp_sell": make_formula({"hetonite_part": 6}, 48 * 6),
        # Heavy Xiranite Assembly: 60 xi + 30 eff -> 6 Heavy Xiranite
        # (previously folded directly into "hx"). Core: hx_sell needs it.
        "hx_make": make_formula({"xi": 60, "eff": 30, "heavy_xiranite": -6}, 0),
        # Sell 6 Heavy Xiranite at $27/unit ($162/multiple, matching hx's
        # original combined output).
        "hx_sell": make_formula({"heavy_xiranite": 6}, 27 * 6),
        # YA: 120 cup -> 6 ya
        "ya": make_formula({"cup": 120}, 22 * 6),
        # YC: 120 ferr -> 6 yc (raw Ferrium Ore, not refined Ferrium --
        # this recipe doesn't need the refined material).
        "yc": make_formula({"ferr": 120}, 16 * 6),
        # Sell xi at $1
        "xi_sell": make_formula({"xi": 1}, 1),
        # Cuprium Part Fitting: 30 cup -> 30 Cuprium Part (previously
        # folded directly into "cp_sell"). Core: cuprium_component needs
        # it too.
        "cuprium_part_make": make_formula({"cup": 30, "cuprium_part": -30}, 0),
        # Sell Cuprium Part at $1 (after cup_conv + cuprium_part_make)
        "cp_sell": make_formula({"cuprium_part": 1}, 1),
    }
    if config.purify_building:
        # Purification Building: 120 inert -> 30 eff
        f["purify"] = make_formula({"inert": 120, "eff": -30}, 0)
    if config.purify_node:
        # Test Area Purification Node: 30 sew -> 1 eff (max 12 multiples)
        f["purify_node"] = make_formula({"sew": 30, "eff": -1}, 0, limit=12)
    if config.secondary_goals:
        # Ferrium Component: 60 Origocrust + 60 Ferrium -> 6 Ferrium
        # Component (the real recipe -- see module docstring).
        f["ferrium_component"] = make_formula({"origocrust": 60, "ferrium": 60}, 0)
        # Origocrust Refining: 30 ori -> 30 Origocrust. Only
        # ferrium_component needs it, so gated alongside it.
        f["origocrust_make"] = make_formula({"ori": 30, "origocrust": -30}, 0)
        # Xiranite Component: 60 Packed Origocrust + 60 Xiranite -> 6
        # Xiranite Component (the real recipe).
        f["xiranite_component"] = make_formula({"packed_origocrust": 60, "xi": 60}, 0)
        # Packed Origocrust Dilution: 60 Origocrust + 30 Sandleaf Powder
        # -> 30 Packed Origocrust (the extra 2:1 dilution step Packed
        # Origocrust needs that plain Origocrust doesn't -- see module
        # docstring). Only xiranite_component needs it.
        f["packed_origocrust_make"] = make_formula(
            {"origocrust": 60, "sandleaf": 30, "packed_origocrust": -30}, 0
        )
        # Cuprium Component: 60 Cuprium Part + 60 Xiranite -> 6 Cuprium
        # Component (the real recipe).
        f["cuprium_component"] = make_formula({"cuprium_part": 60, "xi": 60}, 0)
        # Hetonite Component: 12 Hetonite Part + 12 Heavy Xiranite -> 6
        # Hetonite Component (corrected from the raw recipe list's
        # apparent typo "-> 6 Hetonite Part") -- now an exact 1:1 match
        # to the recipe line instead of a scaled collapsed vector.
        f["hetonite_component"] = make_formula(
            {"hetonite_part": 12, "heavy_xiranite": 12}, 0
        )
        # Sandleaf Powder: Planting Unit (free) + Shredding Unit (30
        # Sandleaf -> 90 Sandleaf Powder) collapsed; consumes none of the
        # tracked *base* resources (matches "very cheap material" in the
        # spec) but produces the "sandleaf" resource dimension ori_to_dop
        # and packed_origocrust_make need. limit is a modest
        # building-count stand-in sized to cover their real floor demand
        # plus the delivery-job target, not a real game constraint.
        f["sandleaf_powder"] = make_formula({"sandleaf": -90}, 0, limit=5)
        # Thermal Bank: 7.5 Originium Ore -> 50 W (tracked via
        # POWER_YIELD, not $ output -- see module docstring). Raw Ore,
        # not dop.
        f["thermal_bank"] = make_formula({"ori": 7.5}, 0)

    for name, limit in config.formula_limits.items():
        if name in f:
            f[name].limit = limit
    for name, output in config.formula_outputs.items():
        if name in f:
            f[name].output = output

    return f


@dataclass
class SearchResult:
    """Result of search(). `all_candidates` is kept so callers can find
    near-optimal discrete (z, metatransfer) branches, e.g. to detect
    discrete-search ties that find_alternatives (an LP-objective-only tool)
    cannot see on its own."""

    result: OptimizeResult
    z: int
    metatransfer: np.ndarray
    formula_names: list[str]
    all_candidates: list[tuple[OptimizeResult, int, np.ndarray]]


def search(config: WulingConfig) -> SearchResult:
    """Search over forge allocations (z) and metatransfer choices, returning
    the best-dollar solution.

    Generalizes tests/wuling/test_wuling_1p2e.py::_search_1p2e.
    """
    formulas = build_formulas(config)
    candidates: list[tuple[OptimizeResult, int, np.ndarray]] = []
    for z in range(config.max_forges + 1):
        if not config.fix_hx_limit:
            formulas["hx_make"].limit = config.max_forges - z
        for mt in config.metatransfers:
            income = config.base_supply + z * XI_PER_FORGE + mt
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, mt.copy()))

    best_result, best_z, best_mt = max(candidates, key=lambda c: c[0].dollar_output)
    return SearchResult(
        result=best_result,
        z=best_z,
        metatransfer=best_mt,
        formula_names=list(formulas.keys()),
        all_candidates=candidates,
    )


def preset_1p2e_full() -> WulingConfig:
    """1.2e full: everything on (the default WulingConfig)."""
    return WulingConfig()


def preset_1p2e_equiv_1p2d() -> WulingConfig:
    """1.2e model with purify_node off, matching 1.2d's base/max_forges."""
    return WulingConfig(purify_node=False)


def preset_1p2_full() -> WulingConfig:
    """1.2e model reproducing the older "1.2 full" base/max_forges."""
    return WulingConfig(
        base_supply=(0, 480, 90, 180) + (0,) * (len(RESOURCE_NAMES) - 4),
        max_forges=8,
        purify_node=False,
    )
