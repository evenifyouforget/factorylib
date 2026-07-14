"""Standard Wuling environment: configurable formula set + forge/metatransfer
search, generalizing the "1.2e full" model (tests/wuling/test_wuling_1p2e.py).

Resource vector (9): [xi, ori, ferr, cup_ore, cup, sew, eff, inert, dop]
Formula order: cup_conv xi_sew ori_to_dop sc lc hp hx ya yc xi_sell cp_sell
               [purify] [purify_node] [ferrium_component xiranite_component
               cuprium_component hetonite_component sandleaf_powder
               thermal_bank]

"dop" (Dense Originium Powder) is its own resource dimension, not folded
into "ori" (raw Originium Ore) equivalent cost as an earlier version of
this module did. SC/LC Wuling Battery actually consume DOP (30 Xircon +
120 DOP -> 6 SC; 30 Xiranite + 90 DOP -> 6 LC), while Ferrium/Xiranite
Component consume Origocrust/Packed Origocrust -- a *different* refining
chain off raw Ore that DOP cannot substitute for, and Thermal Bank burns
raw Ore directly. Folding DOP into a single fungible "ori" pool made a
metatransfer of DOP (see DEFAULT_METATRANSFERS) incorrectly spendable by
the Component/Thermal Bank formulas too. ori_to_dop (60 ori -> 30 dop,
lossless, matching the existing "collapsed" convention of not tracking
the Sandleaf Powder co-input Grinding also needs) lets locally-mined Ore
still fund SC/LC exactly as before; metatransferred dop bypasses that
conversion, exactly as a real metatransfer would. This changes nothing
about $-optimal search() results (ori_to_dop is a lossless pass-through
with no other constraint, so the net effect for local-only play is
identical to the old direct "ori" consumption) -- it only matters once
Components exist to (incorrectly, before this fix) compete for it.

The bracketed "secondary goals" formulas (gated by
WulingConfig.secondary_goals, on by default) exist purely to give the
Part 4/5 fitness function's gear/delivery/power terms something to act
on -- they all have $ output=0, so the raw dollar-maximizing LP in
search() never chooses to run them (any positive rate would only spend
resources the $-formulas are already fully using, at zero marginal $
value), and none of the existing scenario-equivalence tests change. They
are additive collapsed formulas, not full re-derivations of the raw
recipe list, and cover only part of Part 4's goal categories:
  - hetonite_component / cuprium_component / xiranite_component /
    ferrium_component collapse the Gearing Unit recipe plus its whole
    upstream chain (Origocrust, Packed Origocrust, etc. approximated as
    their Originium-Ore-equivalent cost -- the same "collapsed" approach
    already used for sc/lc/hp/hx) into one formula each, scaled from the
    existing hp/hx consumption vectors where those items are inputs.
    Xiranite Component needs 120 ori (not 60): Packed Origocrust has an
    extra 2:1 dilution step (Origocrust Powder -> Dense Origocrust
    Powder) that plain Origocrust (Ferrium Component's input) doesn't
    have. Cryston Component and Amethyst Component are NOT modeled:
    their chains need Amethyst Ore, a base resource this model doesn't
    track at all.
  - sandleaf_powder: Planting Unit + Shredding Unit collapsed into one
    formula that consumes none of the tracked resources (matching the
    spec's "very cheap material" framing) and produces a delivery-job
    filler good. Its limit represents a modest, arbitrary number of
    building instances (this LP has no building-count dimension), not a
    real game constraint.
  - thermal_bank: the simplest Thermal Bank recipe (raw Originium Ore ->
    W), tracked via POWER_YIELD below since Formula.output is $-only.
    The more resource-efficient battery -> power route (spec: "1.5 SC
    Wuling Battery -> 3200 W") is NOT modeled -- it would require
    splitting the existing sc/lc formulas into separate make/sell/power
    steps (as tests/wuling/test_jade_gourd.py already does for hx/hp),
    which risks changing behavior depended on by many existing tests;
    left as a future extension.

Being zero-$, these formulas are also zero-$ *ties* with doing nothing:
find_alternatives (factorylib.alternatives) will correctly report that
e.g. sandleaf_powder's rate is undetermined by the $-maximizing LP (any
value up to its limit is equally optimal at $0 marginal value) -- a real
LP degeneracy, but not an economically meaningful "tied solution" in the
sense Part 2 was designed for (a genuine choice between two strategies).
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

RESOURCE_NAMES = ["xi", "ori", "ferr", "cup_ore", "cup", "sew", "eff", "inert", "dop"]
FORMULA_NAMES = [
    "cup_conv",
    "xi_sew",
    "ori_to_dop",
    "sc",
    "lc",
    "hp",
    "hx",
    "ya",
    "yc",
    "xi_sell",
    "cp_sell",
    "purify",
    "purify_node",
    "ferrium_component",
    "xiranite_component",
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
    "hp": 6.0,
    "hx": 6.0,
    "ya": 6.0,
    "yc": 6.0,
    "xi_sell": 1.0,
    "cp_sell": 1.0,
}

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
}

FORMULA_LABELS = {
    "cup_conv": "Cuprium Ore Refining (Cuprium Ore → Cuprium + Sewage)",
    "xi_sew": "Xiranite + Sewage Reaction (→ Xircon Effluent + Inert Xircon Effluent)",
    "ori_to_dop": "Originium Ore Grinding (→ Dense Originium Powder)",
    "sc": "SC Wuling Battery",
    "lc": "LC Wuling Battery",
    "hp": "Hetonite Part",
    "hx": "Heavy Xiranite",
    "ya": "Yazhen Syringe A",
    "yc": "Yazhen Syringe C",
    "xi_sell": "Xiranite (sold)",
    "cp_sell": "Cuprium Part (sold)",
    "purify": "Purification Building (Inert Xircon Effluent → Xircon Effluent)",
    "purify_node": "Test Area Purification Node (Sewage → Xircon Effluent)",
    "ferrium_component": "Ferrium Component",
    "xiranite_component": "Xiranite Component",
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

DEFAULT_BASE_SUPPLY = (0, 540, 90, 240, 0, 0, 0, 0, 0)  # 1.2e base
DEFAULT_MAX_FORGES = 12
DEFAULT_METATRANSFERS = (
    (0, 0, 0, 0, 0, 0, 0, 0, 25),  # 25 Dense Originium Powder
    (0, 0, 25, 0, 0, 0, 0, 0, 0),  # 25 Ferrium Ore
)
XI_PER_FORGE = np.array([30, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)


@dataclass
class WulingConfig:
    """Standard Wuling environment configuration.

    Defaults reproduce "1.2e full" (everything on) exactly.

    Args:
        base_supply: length-9 resource supply vector (before forge/metatransfer
            top-ups), in RESOURCE_NAMES order.
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
        fix_hx_limit: if True, search() will not override formulas["hx"].limit
            each z iteration (use the configured/overridden limit as-is).
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


def build_formulas(config: WulingConfig) -> dict[str, Formula]:
    """Build a fresh dict of 1.2e-model Formulas per config.

    Ports tests/wuling/test_wuling_1p2e.py::_make_1p2e_formulas (with
    "ori" consumption for SC/LC's DOP portion split out into its own
    "dop" resource via ori_to_dop -- see module docstring), then applies
    config.formula_limits / config.formula_outputs overrides.
    """
    f = {
        # 30 cup_ore -> 30 cup + 30 sew
        "cup_conv": Formula(
            consumption=np.array([0, 0, 0, 30, -30, -30, 0, 0, 0], dtype=float),
            output=0,
        ),
        # 30 xi + 30 sew -> 30 eff + 30 inert
        "xi_sew": Formula(
            consumption=np.array([30, 0, 0, 0, 0, 30, -30, -30, 0], dtype=float),
            output=0,
        ),
        # 60 ori -> 30 dop (collapses 2 Shredding Units, each 30 ori -> 30
        # Originium Powder, feeding 1 Grinding Unit, 60 Powder -> 30 DOP;
        # the Sandleaf Powder co-input Grinding also needs isn't tracked,
        # matching the pre-existing collapsed-formula convention
        # elsewhere). Scaled to belt size (30/min) like every other
        # formula's "1 multiple" -- the minimally-reduced 2:1 ratio alone
        # ("2 ori -> 1 dop") made a formula's rate mean "2 ore/min" here
        # instead of a recognizable physical unit everywhere else.
        "ori_to_dop": Formula(
            consumption=np.array([0, 60, 0, 0, 0, 0, 0, 0, -30], dtype=float),
            output=0,
        ),
        # SC: (60 eff + 30 ferr -> 30 Xircon + 30 sew)
        #     + (30 Xircon + 120 dop -> 6 SC)
        "sc": Formula(
            consumption=np.array([0, 0, 30, 0, 0, -30, 60, 0, 120], dtype=float),
            output=54 * 6,
        ),
        # LC: 30 xi + 90 dop -> 6 LC
        "lc": Formula(
            consumption=np.array([30, 0, 0, 0, 0, 0, 0, 0, 90], dtype=float),
            output=25 * 6,
        ),
        # HP: (240 cup + 30 ferr -> 30 Hetonite + 30 sew) + (30 Hetonite -> 6 HP)
        "hp": Formula(
            consumption=np.array([0, 0, 30, 0, 240, -30, 0, 0, 0], dtype=float),
            output=48 * 6,
        ),
        # HX: 60 xi + 30 eff -> 6 HX
        "hx": Formula(
            consumption=np.array([60, 0, 0, 0, 0, 0, 30, 0, 0], dtype=float),
            output=27 * 6,
        ),
        # YA: 120 cup -> 6 ya
        "ya": Formula(
            consumption=np.array([0, 0, 0, 0, 120, 0, 0, 0, 0], dtype=float),
            output=22 * 6,
        ),
        # YC: 120 ferr -> 6 yc
        "yc": Formula(
            consumption=np.array([0, 0, 120, 0, 0, 0, 0, 0, 0], dtype=float),
            output=16 * 6,
        ),
        # Sell xi at $1
        "xi_sell": Formula(
            consumption=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float), output=1
        ),
        # Sell cup at $1 (after cup_conv conversion)
        "cp_sell": Formula(
            consumption=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float), output=1
        ),
    }
    if config.purify_building:
        # Purification Building: 120 inert -> 30 eff
        f["purify"] = Formula(
            consumption=np.array([0, 0, 0, 0, 0, 0, -30, 120, 0], dtype=float),
            output=0,
        )
    if config.purify_node:
        # Test Area Purification Node: 30 sew -> 1 eff (max 12 multiples)
        f["purify_node"] = Formula(
            consumption=np.array([0, 0, 0, 0, 0, 30, -1, 0, 0], dtype=float),
            output=0,
            limit=12,
        )
    if config.secondary_goals:
        # Ferrium Component: 60 Origocrust + 60 Ferrium -> 6 Ferrium
        # Component. Origocrust/Ferrium collapsed to their Ori/FerrOre
        # equivalent cost (both 1:1 conversions upstream from raw Ore --
        # NOT dop; see module docstring).
        f["ferrium_component"] = Formula(
            consumption=np.array([0, 60, 60, 0, 0, 0, 0, 0, 0], dtype=float), output=0
        )
        # Xiranite Component: 60 Packed Origocrust + 60 Xiranite -> 6
        # Xiranite Component. Packed Origocrust needs 120 ori (not 60):
        # its chain has an extra 2:1 dilution step vs. plain Origocrust.
        f["xiranite_component"] = Formula(
            consumption=np.array([60, 120, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            output=0,
        )
        # Cuprium Component: 60 Cuprium Part + 60 Xiranite -> 6 Cuprium
        # Component. Cuprium Part collapsed to its Cuprium equivalent cost
        # (1:1 via Fitting Unit, same convention as cp_sell above).
        f["cuprium_component"] = Formula(
            consumption=np.array([60, 0, 0, 0, 60, 0, 0, 0, 0], dtype=float), output=0
        )
        # Hetonite Component: 12 Hetonite Part + 12 Heavy Xiranite -> 6
        # Hetonite Component (corrected from the raw recipe list's
        # apparent typo "-> 6 Hetonite Part"). Consumption = 2x hp's +
        # 2x hx's consumption vectors above (12 HP / 12 HX = 2 runs each
        # of those 6-unit-per-run formulas).
        f["hetonite_component"] = Formula(
            consumption=np.array([120, 0, 60, 0, 480, -60, 60, 0, 0], dtype=float),
            output=0,
        )
        # Sandleaf Powder: Planting Unit (free) + Shredding Unit (30
        # Sandleaf -> 90 Sandleaf Powder) collapsed; consumes none of the
        # tracked resources (matches "very cheap material" in the spec).
        # limit is an arbitrary modest building-count stand-in, not a
        # real constraint (this LP has no building-count dimension).
        f["sandleaf_powder"] = Formula(
            consumption=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            output=0,
            limit=10,
        )
        # Thermal Bank: 7.5 Originium Ore -> 50 W (tracked via
        # POWER_YIELD, not $ output -- see module docstring). Raw Ore,
        # not dop.
        f["thermal_bank"] = Formula(
            consumption=np.array([0, 7.5, 0, 0, 0, 0, 0, 0, 0], dtype=float), output=0
        )

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
            formulas["hx"].limit = config.max_forges - z
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
        base_supply=(0, 480, 90, 180, 0, 0, 0, 0, 0),
        max_forges=8,
        purify_node=False,
    )
