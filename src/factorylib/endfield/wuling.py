"""Standard Wuling environment: configurable formula set + forge/metatransfer
search, generalizing the "1.2e full" model (tests/wuling/test_wuling_1p2e.py).

Formulas are built with make_formula(), a name-keyed dict -> consumption
vector helper: it raises immediately on a typo'd resource name instead of
silently writing to the wrong array index (the exact bug class behind
the earlier DOP/Origocrust fungibility bug, where two genuinely
different resources were folded into one index). See RESOURCE_NAMES and
FORMULA_NAMES below for the exact resource/formula ordering.

This module models almost every recipe in factorylib_tmp_problem_statement.md
that doesn't require Amethyst Ore (a base resource this model doesn't
track at all -- Cryston/Amethyst Component and LC Valley Battery are the
only recipes excluded for that reason; Craft Gear is excluded for an
unrelated reason, see below). Nothing is folded into a base-resource-
equivalent cost anymore *except* recipes that only ever consume Water or
Acid alongside a tracked resource -- Water/Acid are themselves free/
unconstrained (Fluid Pump: "(null) -> 60 Water OR 60 Acid"), so a chain
that's otherwise all-Water (e.g. Yazhen/Yazhen Powder/Yazhen Solution)
is folded into one zero-cost formula rather than three, since there's no
scarce resource whose flow that would hide.

Two real DOP/Origocrust-style fungibility hazards this module fixes by
giving each item its own resource dimension:
  - "dop" (Dense Originium Powder) is not folded into "ori" (raw
    Originium Ore): SC/LC Wuling Battery consume DOP, while Ferrium/
    Xiranite Component consume Origocrust/Packed Origocrust -- a
    *different* refining chain off raw Ore that DOP cannot substitute
    for, and Thermal Bank burns raw Ore directly. Folding DOP into a
    single fungible "ori" pool made a metatransfer of DOP (see
    DEFAULT_METATRANSFERS) incorrectly spendable by the Component/
    Thermal Bank formulas too.
  - "ferrium_powder" (Ferrium Powder, from Shredding) is not folded into
    "ferrium" (refined Ferrium, from Refining): xircon_make and
    hetonite_make both need the shredded Powder form specifically (per
    the Reactor Crucible recipes), while ferrium_component/yc/
    ferrium_part_make/ferrium_bottle_make need refined Ferrium directly.
    Numerically harmless either way (both are lossless 1:1 chains off
    raw Ferrium Ore with no other constraint), but conflating them would
    be the same bug class waiting to happen once anything rate-limits
    one form and not the other.

Every "_make"/"_powder_make"/"_solution_make"/"_bottle_make" formula is a
lossless, unconstrained pass-through (limit=inf, no other constraint) --
this changes nothing about $-optimal search() results. Every historical
scenario-equivalence figure (206735/146, 205129/146, 103335/73,
119335/88, 2823/2, 2229/2) is unchanged; uncollapsing only makes each
intermediate resource visible to the fitness/CLI/delivery-job/diagram
layers instead of hidden inside a collapsed coefficient. One genuine
behavior change from uncollapsing: jincao_tea (see below) is a perfect
economic substitute for ya (identical recipe shape, identical price), so
banning ya alone no longer reduces $-optimal output the way it used to --
banning jincao_tea too is required to reproduce that historical figure
(see test_formula_limit_override_ban_ya).

Yazhen Syringe A/C and the newly-added Jincao Tea/Drink now route through
the real recipes instead of a raw-ore-equivalent shortcut: "ya"/"yc"/
"jincao_tea"/"jincao_drink" each consume 60 [Cuprium|Ferrium] Part + 30
[Cuprium|Ferrium] Bottle + 30 [Yazhen|Jincao] Solution (the Filling Unit
step -- "30 Bottle + 30 fluid -> 30 filled Bottle" -- is folded directly
into these four formulas rather than tracked as its own resource, since
nothing else ever needs a "filled Bottle" independently; same reasoning
already applied to Water-only chains). Yazhen/Jincao Solution production
(Planting -> Shredding -> Reactor Crucible, entirely Water-fed) is folded
into yazhen_solution_make/jincao_solution_make for the same reason.

SC/HC Valley Battery are new, entirely zero-$ (there's no Sell recipe for
either in the spec -- unlike SC/LC *Wuling* Battery, which do sell) --
their only real purpose is feeding Thermal Bank's more efficient battery
-> power routes ("1.5 SC Valley Battery -> 420 W", "1.5 HC Valley Battery
-> 1100 W"), so sc_valley/hc_valley and their thermal_bank_sc_valley/
thermal_bank_hc_valley consumers are secondary-goal formulas like
thermal_bank itself. LC Valley Battery is excluded (needs Amethyst Part).
HC Valley Battery's chain (Steel <- Dense Ferrium Powder <- Ferrium
Powder + Sandleaf Powder) doesn't need Amethyst, so it's included.

Craft Gear ("8000 Wuling Stock Bill + 50 Xiranite Component -> 1 Xiranite
Component Gear", etc.) is NOT modeled as a Formula: it spends *accumulated*
Stock Bill (a one-time stock), not a steady per-minute flow, which doesn't
fit this LP's steady-state framework at all (every other formula converts
one per-minute rate into another; Craft Gear converts a lump of savings
into a one-time item). Not modeled at all currently -- Components already
show their own /min rate directly in the CLI's formula listing.

The bracketed "secondary goals" formulas (gated by
WulingConfig.secondary_goals, on by default) exist purely to give the
Part 4/5 fitness function's gear/delivery/power terms something to act
on -- they all have $ output=0, so the raw dollar-maximizing LP in
search() never chooses to run them (any positive rate would only spend
resources the $-formulas are already fully using, at zero marginal $
value), and none of the existing scenario-equivalence tests change:
  - hetonite_component / cuprium_component / xiranite_component /
    ferrium_component consume the real intermediates directly (e.g.
    hetonite_component is an exact 1:1 match to the recipe line: 12
    Hetonite Part + 12 Heavy Xiranite -> 6 Hetonite Component, corrected
    from the raw recipe list's apparent typo "-> 6 Hetonite Part").
    origocrust_make/packed_origocrust_make are gated alongside
    ferrium_component/xiranite_component since nothing else consumes
    Origocrust/Packed Origocrust. Cryston Component and Amethyst
    Component are still NOT modeled (need Amethyst Ore).
  - sandleaf_powder (Shredding Unit: 30 Sandleaf -> 90 Sandleaf Powder)
    is fed by sandleaf_plant (Planting Unit: -> 30 Sandleaf, gated
    alongside it since nothing else needs raw Sandleaf); sandleaf_powder
    itself consumes none of the tracked *base* resources (matching the
    spec's "very cheap material" framing) but does produce the
    "sandleaf" resource dimension ori_to_dop/packed_origocrust_make/
    dense_ferrium_powder_make need. sandleaf_plant's limit (5) is a
    modest, arbitrary building-count stand-in (this LP has no
    building-count dimension), sized to comfortably cover its tracked
    consumers' floor demand plus the delivery-job target.
  - thermal_bank (raw Originium Ore -> W) / thermal_bank_sc_valley /
    thermal_bank_hc_valley / sc_power / lc_power are the five power
    routes tracked via POWER_YIELD below, since Formula.output is
    $-only. sc_power/lc_power (the spec's most resource-efficient
    route: "1.5 SC Wuling Battery -> 3200 W", "1.5 LC Wuling Battery ->
    1600 W") consume the *same* sc_battery/lc_battery pool sc_sell/
    lc_sell draw from (see sc_make's comment) -- a battery burned for
    power is one that can't also be sold, a real mutual-exclusivity
    constraint, not a free byproduct.

Being zero-$, these formulas are also zero-$ *ties* with doing nothing,
above whatever floor the $-maximizing LP actually needs from them.
sandleaf_powder is the one exception with a genuine floor now: its
tracked consumers need real amounts of "sandleaf" at their own LP-chosen
rates, so sandleaf_powder's rate is pinned to at least that floor -- but
any excess above it remains a real LP degeneracy (any value up to its
limit is equally optimal at $0 marginal value), same as every other
secondary-goal formula always is. Not an economically meaningful "tied
solution" in the sense Part 2 was designed for (a genuine choice between
two strategies). factorylib.endfield.cli filters SECONDARY_GOAL_FORMULA_NAMES
(plus the plumbing formulas that solely feed them) out of its
tied-alternatives search for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factorylib.optimize import Formula, OptimizeResult, maximize_dollar

# Formulas that exist only to give the Part 4/5 fitness function
# something to act on (see module docstring); all have $ output=0. Note:
# despite being here, sandleaf_powder must NOT be dropped from cli.py's
# tie-detection LP -- see SECONDARY_PLUMBING_FORMULA_NAMES's comment.
SECONDARY_GOAL_FORMULA_NAMES = (
    "ferrium_component",
    "xiranite_component",
    "cuprium_component",
    "hetonite_component",
    "sandleaf_powder",
    "thermal_bank",
    "thermal_bank_sc_valley",
    "thermal_bank_hc_valley",
    "sc_power",
    "lc_power",
)

# Plumbing "_make" formulas that exist solely to feed a
# SECONDARY_GOAL_FORMULA_NAMES formula and nothing *core* -- excluded from
# cli.py's tied-alternatives search alongside SECONDARY_GOAL_FORMULA_NAMES:
# their sole consumer already being excluded would otherwise leave them
# as dangling zero-$ formulas that find_alternatives' epsilon nudge could
# "discover" spurious, resource-wasting alternatives around (see cli.py).
# sandleaf_plant is deliberately NOT here even though it only feeds
# sandleaf_powder: sandleaf_powder itself is excluded from this list too
# (see SECONDARY_GOAL_FORMULA_NAMES's own note) because ori_to_dop -- a
# CORE, always-present formula -- genuinely needs sandleaf_powder's
# output, so removing sandleaf_powder/sandleaf_plant from the
# tie-detection LP entirely would starve ori_to_dop and corrupt the
# *baseline* dollar figure within that filtered sub-problem, not just
# suppress a spurious alternative.
SECONDARY_PLUMBING_FORMULA_NAMES = (
    "origocrust_make",
    "packed_origocrust_make",
    "dense_ferrium_powder_make",
    "steel_make",
    "steel_part_make",
    "sc_valley",
    "hc_valley",
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
    "originium_powder",
    "sandleaf_raw",
    "ferrium_powder",
    "liquid_xiranite",
    "cuprium_powder",
    "cuprium_solution",
    "hetonite_solution",
    "ferrium_part",
    "cuprium_bottle",
    "ferrium_bottle",
    "yazhen_solution",
    "jincao_solution",
    "dense_ferrium_powder",
    "steel",
    "steel_part",
    "sc_valley_battery",
    "hc_valley_battery",
    "sc_battery",
    "lc_battery",
    # Virtual bookkeeping resources (no physical belt/pipe flow -- see
    # build_formulas's module docstring): forge_budget lets
    # xiranite_forge_alloc/heavy_xiranite_forge_alloc compete for the
    # same max_forges pool; hx_forge_capacity lets
    # heavy_xiranite_forge_alloc cap hx_make's rate; metatransfer_
    # allowance lets the metatransfer-option formulas compete for the
    # single "pick exactly one" choice. All three turn what used to be an
    # external brute-force loop over (z, metatransfer) into ordinary
    # Formula entries solved by the same MILP as everything else.
    "forge_budget",
    "hx_forge_capacity",
    "metatransfer_allowance",
]
FORMULA_NAMES = [
    "cup_conv",
    "originium_powder_make",
    "ori_to_dop",
    "ferrium_make",
    "ferrium_powder_make",
    "liquid_xiranite_make",
    "xi_sew",
    "xircon_make",
    "sc_make",
    "sc_sell",
    "sc_power",
    "lc_make",
    "lc_sell",
    "lc_power",
    "cuprium_powder_make",
    "cuprium_solution_make",
    "hetonite_solution_make",
    "hetonite_make",
    "hp_make",
    "hp_sell",
    "hx_make",
    "hx_sell",
    "ferrium_part_make",
    "cuprium_bottle_make",
    "ferrium_bottle_make",
    "yazhen_solution_make",
    "jincao_solution_make",
    "ya",
    "yc",
    "jincao_tea",
    "jincao_drink",
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
    "sandleaf_plant",
    "sandleaf_powder",
    "thermal_bank",
    "dense_ferrium_powder_make",
    "steel_make",
    "steel_part_make",
    "sc_valley",
    "hc_valley",
    "thermal_bank_sc_valley",
    "thermal_bank_hc_valley",
]

# Belts (solids) run at 30 items/min; pipes (liquids) run at 120 items/min
# (see factorylib_tmp_physical_factory_construction.md). Used to price a
# resource flow's physical complexity in terms of "how many belts/pipes"
# it represents, rather than the abstract recipe-multiple fraction alone
# -- see factorylib.endfield.goals.fitness. Liquid/pipe classification for
# the "Solution"/"Liquid X" items is inferred from naming convention
# (matching Sewage/Xircon Effluent, the model's other Reactor-Crucible-
# produced fluids), not separately confirmed in-game.
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
    "originium_powder": 30.0,
    "sandleaf_raw": 30.0,
    "ferrium_powder": 30.0,
    "liquid_xiranite": 120.0,
    "cuprium_powder": 30.0,
    "cuprium_solution": 120.0,
    "hetonite_solution": 120.0,
    "ferrium_part": 30.0,
    "cuprium_bottle": 30.0,
    "ferrium_bottle": 30.0,
    "yazhen_solution": 120.0,
    "jincao_solution": 120.0,
    "dense_ferrium_powder": 30.0,
    "steel": 30.0,
    "steel_part": 30.0,
    "sc_valley_battery": 30.0,
    "hc_valley_battery": 30.0,
    "sc_battery": 30.0,
    "lc_battery": 30.0,
}


def is_forge_or_metatransfer_formula(name: str) -> bool:
    """True for the forge-allocation/metatransfer-choice bookkeeping
    formulas (see build_formulas's module docstring) --
    factorylib.endfield.goals._plan_complexity excludes them from the
    fraction-complexity penalty entirely: xiranite_forge_alloc/
    heavy_xiranite_forge_alloc's real Xiranite flow is always an exact
    integer number of belts by construction (1 forge == 1 belt's worth),
    so excluding it never changes anything numerically, but a
    metatransfer_option_* formula's real resource flow (e.g. 25 Dense
    Originium Powder) generally is *not* a clean belt multiple -- and
    Metatransfer is a discrete inventory action, not a continuous belt,
    so pricing it as one would be pricing something that was never
    priced before this refactor and isn't a real physical throughput
    concern at all."""
    return name in ("xiranite_forge_alloc", "heavy_xiranite_forge_alloc") or (
        name.startswith("metatransfer_option_")
    )


# W produced per multiple of a formula's rate. Formula.output is $-only, so
# a formula that only contributes power (no $ value) is tracked here
# instead -- see plan_from_search_result in factorylib.endfield.goals.
POWER_YIELD = {
    "thermal_bank": 50.0,
    "thermal_bank_sc_valley": 420.0,
    "thermal_bank_hc_valley": 1100.0,
    "sc_power": 3200.0,
    "lc_power": 1600.0,
}

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
    "sc_sell": 6.0,
    "lc_sell": 6.0,
    "hp_sell": 6.0,
    "hx_sell": 6.0,
    "ya": 6.0,
    "yc": 6.0,
    "jincao_tea": 6.0,
    "jincao_drink": 6.0,
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
# Wuling Battery, Xiranite/Cuprium Part sold, Jincao Tea/Drink) has no
# such upstream dependency, so it's lowest priority by default -- see
# factorylib.priority_sell.allocate_by_priority.
SELL_PRIORITY = ("ya", "hp_sell", "hx_sell", "sc_sell", "yc")

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
    "originium_powder": "Originium Powder",
    "sandleaf_raw": "Sandleaf",
    "ferrium_powder": "Ferrium Powder",
    "liquid_xiranite": "Liquid Xiranite",
    "cuprium_powder": "Cuprium Powder",
    "cuprium_solution": "Cuprium Solution",
    "hetonite_solution": "Hetonite Solution",
    "ferrium_part": "Ferrium Part",
    "cuprium_bottle": "Cuprium Bottle",
    "ferrium_bottle": "Ferrium Bottle",
    "yazhen_solution": "Yazhen Solution",
    "jincao_solution": "Jincao Solution",
    "dense_ferrium_powder": "Dense Ferrium Powder",
    "steel": "Steel",
    "steel_part": "Steel Part",
    "sc_valley_battery": "SC Valley Battery",
    "hc_valley_battery": "HC Valley Battery",
    "sc_battery": "SC Wuling Battery",
    "lc_battery": "LC Wuling Battery",
    "forge_budget": "Forge of the Sky Budget",
    "hx_forge_capacity": "Heavy Xiranite Forge Capacity",
    "metatransfer_allowance": "Metatransfer Allowance",
}

FORMULA_LABELS = {
    "cup_conv": "Cuprium Ore Refining (Cuprium Ore → Cuprium + Sewage)",
    "originium_powder_make": (
        "Originium Powder Shredding (Originium Ore → Originium Powder)"
    ),
    "xi_sew": "Xiranite + Sewage Reaction (→ Xircon Effluent + Inert Xircon Effluent)",
    "ori_to_dop": "Originium Powder Grinding (→ Dense Originium Powder)",
    "ferrium_make": "Ferrium Refining (Ferrium Ore → Ferrium)",
    "ferrium_powder_make": "Ferrium Powder Shredding (Ferrium → Ferrium Powder)",
    "liquid_xiranite_make": "Xiranite Reaction (Xiranite → Liquid Xiranite)",
    "xircon_make": (
        "Xircon Reaction (Xircon Effluent + Ferrium Powder → Xircon + Sewage)"
    ),
    "sc_make": (
        "SC Wuling Battery Packaging (Xircon + Dense Originium Powder → "
        "SC Wuling Battery)"
    ),
    "sc_sell": "SC Wuling Battery (sellable)",
    "sc_power": "Thermal Bank (SC Wuling Battery → Power)",
    "lc_make": (
        "LC Wuling Battery Packaging (Xiranite + Dense Originium Powder → "
        "LC Wuling Battery)"
    ),
    "lc_sell": "LC Wuling Battery (sellable)",
    "lc_power": "Thermal Bank (LC Wuling Battery → Power)",
    "cuprium_powder_make": "Cuprium Powder Shredding (Cuprium → Cuprium Powder)",
    "cuprium_solution_make": (
        "Cuprium Solution Reaction (Cuprium Powder → Cuprium Solution)"
    ),
    "hetonite_solution_make": (
        "Hetonite Solution Purification (Cuprium Solution → Hetonite Solution)"
    ),
    "hetonite_make": (
        "Hetonite Reaction (Hetonite Solution + Ferrium Powder → Hetonite + Sewage)"
    ),
    "hp_make": "Hetonite Part Fitting (Hetonite → Hetonite Part)",
    "hp_sell": "Hetonite Part (sellable)",
    "hx_make": "Heavy Xiranite Forge (Xiranite + Xircon Effluent → Heavy Xiranite)",
    "hx_sell": "Heavy Xiranite (sellable)",
    "ferrium_part_make": "Ferrium Part Fitting (Ferrium → Ferrium Part)",
    "cuprium_bottle_make": "Cuprium Bottle Moulding (Cuprium → Cuprium Bottle)",
    "ferrium_bottle_make": "Ferrium Bottle Moulding (Ferrium → Ferrium Bottle)",
    "yazhen_solution_make": "Yazhen Solution (Planting + Shredding + Reactor Crucible)",
    "jincao_solution_make": "Jincao Solution (Planting + Shredding + Reactor Crucible)",
    "ya": "Yazhen Syringe A (sellable)",
    "yc": "Yazhen Syringe C (sellable)",
    "jincao_tea": "Jincao Tea (sellable)",
    "jincao_drink": "Jincao Drink (sellable)",
    "xi_sell": "Xiranite (sellable)",
    "cuprium_part_make": "Cuprium Part Fitting (Cuprium → Cuprium Part)",
    "cp_sell": "Cuprium Part (sellable)",
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
    "sandleaf_plant": "Sandleaf Planting (→ Sandleaf)",
    "sandleaf_powder": "Sandleaf Powder",
    "thermal_bank": "Thermal Bank (Originium Ore → Power)",
    "dense_ferrium_powder_make": (
        "Dense Ferrium Powder Grinding (Ferrium Powder + Sandleaf Powder → "
        "Dense Ferrium Powder)"
    ),
    "steel_make": "Steel Refining (Dense Ferrium Powder → Steel)",
    "steel_part_make": "Steel Part Fitting (Steel → Steel Part)",
    "sc_valley": (
        "SC Valley Battery (Ferrium Part + Originium Powder → SC Valley Battery)"
    ),
    "hc_valley": (
        "HC Valley Battery (Steel Part + Dense Originium Powder → HC Valley Battery)"
    ),
    "thermal_bank_sc_valley": "Thermal Bank (SC Valley Battery → Power)",
    "thermal_bank_hc_valley": "Thermal Bank (HC Valley Battery → Power)",
    "xiranite_forge_alloc": "Forge of the Sky (→ Xiranite supply)",
    "heavy_xiranite_forge_alloc": "Forge of the Sky (→ Heavy Xiranite capacity)",
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
            and their supporting plumbing (Components, Sandleaf Powder,
            Thermal Bank plus SC/HC Valley Battery's power route -- see
            module docstring) are included. They never affect $-optimal
            search() results (zero $ output), only what the Part 4/5
            fitness function and refine() can act on.
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
    consumption: dict[str, float],
    output: float,
    limit: float = np.inf,
    integer: bool = False,
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
    return Formula(consumption=vec, output=output, limit=limit, integer=integer)


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
    # hx_make's rate is capped by however many Forge of the Sky units get
    # allocated to Heavy Xiranite capacity (see heavy_xiranite_forge_alloc
    # below) via a shared "hx_forge_capacity" resource, instead of an
    # external `.limit = max_forges - z` override recomputed by every
    # caller. fix_hx_limit=True keeps the old behavior (a static limit,
    # not tied to any forge allocation at all -- see WulingConfig's
    # docstring) by simply not metering this consumption.
    _hx_make_capacity = {} if config.fix_hx_limit else {"hx_forge_capacity": 1.0}
    f = {
        # 30 cup_ore -> 30 cup + 30 sew
        "cup_conv": make_formula({"cup_ore": 30, "cup": -30, "sew": -30}, 0),
        # Shredding Unit: 30 Originium Ore -> 30 Originium Powder.
        "originium_powder_make": make_formula({"ori": 30, "originium_powder": -30}, 0),
        # Grinding Unit: 60 Originium Powder (+ 30 sandleaf, if
        # secondary_goals) -> 30 Dense Originium Powder. Scaled to belt
        # size (30/min) like every other formula's "1 multiple" -- the
        # minimally-reduced 2:1 ratio alone ("2 Powder -> 1 DOP") made a
        # formula's rate mean "2 Powder/min" here instead of a
        # recognizable physical unit everywhere else.
        "ori_to_dop": make_formula(
            {"originium_powder": 60, "dop": -30, "sandleaf": _ori_to_dop_sandleaf}, 0
        ),
        # Refining Unit: 30 Ferrium Ore -> 30 Ferrium. Core (not
        # secondary-gated): the Xircon/Hetonite reactions below, yc, and
        # ferrium_component all need it.
        "ferrium_make": make_formula({"ferr": 30, "ferrium": -30}, 0),
        # Shredding Unit: 30 Ferrium -> 30 Ferrium Powder. Distinct from
        # "ferrium" itself -- xircon_make/hetonite_make need the shredded
        # Powder form specifically (per the real Reactor Crucible
        # recipes), not refined Ferrium directly (see module docstring).
        "ferrium_powder_make": make_formula({"ferrium": 30, "ferrium_powder": -30}, 0),
        # Reactor Crucible: 30 Xiranite + 30 Water -> 30 Liquid Xiranite
        # (Water free/unconstrained, matching every other Water-fed step).
        "liquid_xiranite_make": make_formula({"xi": 30, "liquid_xiranite": -30}, 0),
        # Reactor Crucible: 30 Liquid Xiranite + 30 Sewage -> 30 Xircon
        # Effluent + 30 Inert Xircon Effluent.
        "xi_sew": make_formula(
            {"liquid_xiranite": 30, "sew": 30, "eff": -30, "inert": -30}, 0
        ),
        # Reactor Crucible: 60 Xircon Effluent + 30 Ferrium Powder -> 30
        # Xircon + 30 Sewage.
        "xircon_make": make_formula(
            {"eff": 60, "ferrium_powder": 30, "xircon": -30, "sew": -30}, 0
        ),
        # Packaging Unit: 30 Xircon + 120 Dense Originium Powder -> 6 SC
        # Wuling Battery. Its own resource (not a direct $ output): unlike
        # Hetonite Part/Heavy Xiranite, SC/LC Wuling Battery have a
        # *second* real consumer (Thermal Bank's battery -> power route,
        # see sc_power/lc_power below) competing for the same physical
        # batteries -- the same battery can't be both sold and burned for
        # power, so it must be a shared resource, not a direct output.
        "sc_make": make_formula({"xircon": 30, "dop": 120, "sc_battery": -6}, 0),
        # Sell 6 SC Wuling Battery at $54/unit ($324/multiple).
        "sc_sell": make_formula({"sc_battery": 6}, 54 * 6),
        # Packaging Unit: 30 Xiranite + 90 Dense Originium Powder -> 6 LC
        # Wuling Battery. Same reasoning as sc_make above.
        "lc_make": make_formula({"xi": 30, "dop": 90, "lc_battery": -6}, 0),
        # Sell 6 LC Wuling Battery at $25/unit ($150/multiple).
        "lc_sell": make_formula({"lc_battery": 6}, 25 * 6),
        # Shredding Unit: 30 Cuprium -> 30 Cuprium Powder.
        "cuprium_powder_make": make_formula({"cup": 30, "cuprium_powder": -30}, 0),
        # Reactor Crucible: 30 Cuprium Powder + 30 Acid -> 30 Cuprium
        # Solution (Acid free/unconstrained).
        "cuprium_solution_make": make_formula(
            {"cuprium_powder": 30, "cuprium_solution": -30}, 0
        ),
        # Purification Unit: 120 Cuprium Solution -> 30 Hetonite Solution
        # + 30 Acid (Acid free byproduct, not tracked).
        "hetonite_solution_make": make_formula(
            {"cuprium_solution": 120, "hetonite_solution": -30}, 0
        ),
        # Reactor Crucible: 60 Hetonite Solution + 30 Ferrium Powder -> 30
        # Hetonite + 30 Sewage. Core: hp_sell needs it.
        "hetonite_make": make_formula(
            {
                "hetonite_solution": 60,
                "ferrium_powder": 30,
                "hetonite": -30,
                "sew": -30,
            },
            0,
        ),
        # Fitting Unit: 30 Hetonite -> 6 Hetonite Part.
        "hp_make": make_formula({"hetonite": 30, "hetonite_part": -6}, 0),
        # Sell 6 Hetonite Part at $48/unit ($288/multiple).
        "hp_sell": make_formula({"hetonite_part": 6}, 48 * 6),
        # Forge of the Sky: 60 Xiranite + 30 Xircon Effluent -> 6 Heavy
        # Xiranite. Core: hx_sell needs it. Rate capped via
        # hx_forge_capacity (see _hx_make_capacity above), not a static
        # limit -- its own limit is left unbounded.
        "hx_make": make_formula(
            {"xi": 60, "eff": 30, "heavy_xiranite": -6, **_hx_make_capacity}, 0
        ),
        # Sell 6 Heavy Xiranite at $27/unit ($162/multiple).
        "hx_sell": make_formula({"heavy_xiranite": 6}, 27 * 6),
        # Fitting Unit: 30 Ferrium -> 30 Ferrium Part. Distinct from
        # "ferrium" -- needed by yc/sc_valley, not ferrium_component
        # (whose real recipe uses refined Ferrium directly).
        "ferrium_part_make": make_formula({"ferrium": 30, "ferrium_part": -30}, 0),
        # Moulding Unit: 60 Cuprium -> 30 Cuprium Bottle.
        "cuprium_bottle_make": make_formula({"cup": 60, "cuprium_bottle": -30}, 0),
        # Moulding Unit: 60 Ferrium -> 30 Ferrium Bottle.
        "ferrium_bottle_make": make_formula({"ferrium": 60, "ferrium_bottle": -30}, 0),
        # Planting Unit (30 Water -> 60 Yazhen) + Shredding Unit (30
        # Yazhen -> 60 Yazhen Powder) + Reactor Crucible (30 Yazhen
        # Powder + 30 Water -> 30 Yazhen Solution) collapsed into one
        # formula: unlike Sandleaf, nothing else in this model needs raw
        # Yazhen/Yazhen Powder independently, and the whole chain is
        # Water-fed (free/unconstrained), so there's no scarce resource
        # flow this would hide.
        "yazhen_solution_make": make_formula({"yazhen_solution": -30}, 0),
        # Same as yazhen_solution_make, for Jincao instead of Yazhen.
        "jincao_solution_make": make_formula({"jincao_solution": -30}, 0),
        # Packaging Unit: 60 Cuprium Part + 30 Cuprium Bottle filled with
        # Yazhen Solution -> 6 Yazhen Syringe A. The Filling Unit step
        # ("30 Bottle + 30 fluid -> 30 filled Bottle") is folded directly
        # in here rather than tracked as its own resource, since nothing
        # else ever needs a "filled Bottle" independently.
        "ya": make_formula(
            {"cuprium_part": 60, "cuprium_bottle": 30, "yazhen_solution": 30}, 22 * 6
        ),
        # Packaging Unit: 60 Ferrium Part + 30 Ferrium Bottle filled with
        # Yazhen Solution -> 6 Yazhen Syringe C.
        "yc": make_formula(
            {"ferrium_part": 60, "ferrium_bottle": 30, "yazhen_solution": 30}, 16 * 6
        ),
        # Packaging Unit: 60 Cuprium Part + 30 Cuprium Bottle filled with
        # Jincao Solution -> 6 Jincao Tea. Same price/shape as "ya" --
        # a perfect economic substitute (see module docstring).
        "jincao_tea": make_formula(
            {"cuprium_part": 60, "cuprium_bottle": 30, "jincao_solution": 30}, 22 * 6
        ),
        # Packaging Unit: 60 Ferrium Part + 30 Ferrium Bottle filled with
        # Jincao Solution -> 6 Jincao Drink. Same price/shape as "yc".
        "jincao_drink": make_formula(
            {"ferrium_part": 60, "ferrium_bottle": 30, "jincao_solution": 30}, 16 * 6
        ),
        # Sell xi at $1
        "xi_sell": make_formula({"xi": 1}, 1),
        # Fitting Unit: 30 Cuprium -> 30 Cuprium Part. Core: cp_sell,
        # ya/jincao_tea, and cuprium_component all need it.
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
        # Gearing Unit: 60 Origocrust + 60 Ferrium -> 6 Ferrium Component
        # (the real recipe -- see module docstring).
        f["ferrium_component"] = make_formula({"origocrust": 60, "ferrium": 60}, 0)
        # Refining Unit: 30 Originium Ore -> 30 Origocrust. Only
        # ferrium_component needs it, so gated alongside it.
        f["origocrust_make"] = make_formula({"ori": 30, "origocrust": -30}, 0)
        # Gearing Unit: 60 Packed Origocrust + 60 Xiranite -> 6 Xiranite
        # Component (the real recipe).
        f["xiranite_component"] = make_formula({"packed_origocrust": 60, "xi": 60}, 0)
        # Refining Unit (Origocrust Powder -> Dense Origocrust Powder via
        # Grinding Unit) + Refining Unit (-> Packed Origocrust) collapsed:
        # 60 Origocrust + 30 Sandleaf Powder -> 30 Packed Origocrust (the
        # extra 2:1 dilution step Packed Origocrust needs that plain
        # Origocrust doesn't -- see module docstring). Only
        # xiranite_component needs it.
        f["packed_origocrust_make"] = make_formula(
            {"origocrust": 60, "sandleaf": 30, "packed_origocrust": -30}, 0
        )
        # Gearing Unit: 60 Cuprium Part + 60 Xiranite -> 6 Cuprium
        # Component (the real recipe).
        f["cuprium_component"] = make_formula({"cuprium_part": 60, "xi": 60}, 0)
        # Gearing Unit: 12 Hetonite Part + 12 Heavy Xiranite -> 6
        # Hetonite Component (corrected from the raw recipe list's
        # apparent typo "-> 6 Hetonite Part") -- an exact 1:1 match to
        # the recipe line.
        f["hetonite_component"] = make_formula(
            {"hetonite_part": 12, "heavy_xiranite": 12}, 0
        )
        # Planting Unit: (null) -> 30 Sandleaf. Free (no tracked base
        # resource cost), gated since nothing else needs raw Sandleaf.
        # limit is a modest, arbitrary building-count stand-in (this LP
        # has no building-count dimension), not a real game constraint.
        f["sandleaf_plant"] = make_formula({"sandleaf_raw": -30}, 0, limit=5)
        # Shredding Unit: 30 Sandleaf -> 90 Sandleaf Powder. Produces the
        # "sandleaf" resource dimension ori_to_dop/packed_origocrust_make
        # /dense_ferrium_powder_make need.
        f["sandleaf_powder"] = make_formula({"sandleaf_raw": 30, "sandleaf": -90}, 0)
        # Thermal Bank: 7.5 Originium Ore -> 50 W (tracked via
        # POWER_YIELD, not $ output -- see module docstring). Raw Ore,
        # not dop.
        f["thermal_bank"] = make_formula({"ori": 7.5}, 0)
        # Thermal Bank: 1.5 SC Wuling Battery -> 3200 W. Competes with
        # sc_sell for the same sc_battery pool (see sc_make's comment) --
        # the spec's most resource-efficient power route, but it means
        # every battery burned for power is one *not* sold.
        f["sc_power"] = make_formula({"sc_battery": 1.5}, 0)
        # Thermal Bank: 1.5 LC Wuling Battery -> 1600 W. Same tradeoff.
        f["lc_power"] = make_formula({"lc_battery": 1.5}, 0)
        # Grinding Unit: 60 Ferrium Powder + 30 Sandleaf Powder -> 30
        # Dense Ferrium Powder. Only feeds the Steel/HC Valley Battery
        # chain, so gated alongside it.
        f["dense_ferrium_powder_make"] = make_formula(
            {"ferrium_powder": 60, "sandleaf": 30, "dense_ferrium_powder": -30}, 0
        )
        # Refining Unit: 30 Dense Ferrium Powder -> 30 Steel.
        f["steel_make"] = make_formula({"dense_ferrium_powder": 30, "steel": -30}, 0)
        # Fitting Unit: 30 Steel -> 30 Steel Part.
        f["steel_part_make"] = make_formula({"steel": 30, "steel_part": -30}, 0)
        # Packaging Unit: 60 Ferrium Part + 90 Originium Powder -> 6 SC
        # Valley Battery. Zero $ (no Sell recipe for it, unlike SC
        # *Wuling* Battery) -- its only purpose is feeding Thermal Bank's
        # more efficient battery -> power route below.
        f["sc_valley"] = make_formula(
            {"ferrium_part": 60, "originium_powder": 90, "sc_valley_battery": -6}, 0
        )
        # Packaging Unit: 60 Steel Part + 90 Dense Originium Powder -> 6
        # HC Valley Battery. Also zero $, same reason. LC Valley Battery
        # is NOT modeled: it needs Amethyst Part, which needs Amethyst
        # Ore (a base resource this model doesn't track at all).
        f["hc_valley"] = make_formula(
            {"steel_part": 60, "dop": 90, "hc_valley_battery": -6}, 0
        )
        # Thermal Bank: 1.5 SC Valley Battery -> 420 W.
        f["thermal_bank_sc_valley"] = make_formula({"sc_valley_battery": 1.5}, 0)
        # Thermal Bank: 1.5 HC Valley Battery -> 1100 W.
        f["thermal_bank_hc_valley"] = make_formula({"hc_valley_battery": 1.5}, 0)

    # Forge of the Sky allocation, as ordinary (integer) Formula entries
    # instead of an external brute-force loop over z -- both compete for
    # the same max_forges "forge_budget" pool (see full_supply()).
    f["xiranite_forge_alloc"] = make_formula(
        {"forge_budget": 1.0, "xi": -30.0},
        0,
        limit=float(config.max_forges),
        integer=True,
    )
    if not config.fix_hx_limit:
        # -> 1 hx_forge_capacity, which hx_make now consumes 1-per-
        # multiple (see _hx_make_capacity above).
        f["heavy_xiranite_forge_alloc"] = make_formula(
            {"forge_budget": 1.0, "hx_forge_capacity": -1.0},
            0,
            limit=float(config.max_forges),
            integer=True,
        )

    # Metatransfer choice, as one competing integer Formula per option --
    # each produces that option's resource vector directly, and all of
    # them draw on the same "pick exactly one" metatransfer_allowance
    # pool (see full_supply()).
    for i, mt in enumerate(config.metatransfers):
        vec = np.zeros(len(RESOURCE_NAMES), dtype=float)
        vec[RESOURCE_NAMES.index("metatransfer_allowance")] = 1.0
        vec -= np.asarray(mt, dtype=float)
        f[f"metatransfer_option_{i}"] = Formula(
            consumption=vec, output=0.0, limit=1.0, integer=True
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
    """Result of search(). z/metatransfer are derived from the solved
    rates of the xiranite_forge_alloc/heavy_xiranite_forge_alloc/
    metatransfer_option_* formulas (see build_formulas), kept as their
    own fields since callers display them specially (_format_forge_
    allocation, _format_metatransfer) rather than as generic formula
    rates."""

    result: OptimizeResult
    z: int
    metatransfer: np.ndarray
    formula_names: list[str]


def full_supply(config: WulingConfig) -> np.ndarray:
    """The complete resource-supply vector search() (and any other
    direct maximize_dollar caller working with build_formulas' output)
    should use: config.base_supply plus the fixed amounts the forge/
    metatransfer choice formulas compete over (see build_formulas's
    module docstring) -- max_forges of forge_budget, and (if any
    metatransfer options exist) exactly 1 metatransfer_allowance."""
    supply = config.base_supply.copy()
    supply[RESOURCE_NAMES.index("forge_budget")] += config.max_forges
    if config.metatransfers:
        supply[RESOURCE_NAMES.index("metatransfer_allowance")] += 1.0
    return supply


def search(config: WulingConfig) -> SearchResult:
    """Find the $-optimal production plan, including which discrete
    Forge of the Sky allocation (Xiranite supply vs. Heavy Xiranite
    capacity) and Metatransfer option to pick -- all solved together in
    one MILP (see factorylib.optimize.maximize_dollar's docstring on the
    integer-formula path), rather than the previous brute-force loop
    over every (z, metatransfer) combination. A nice side effect:
    factorylib.alternatives.find_alternatives' epsilon-perturbation tie
    finder now surfaces ties *between* discrete choices for free, the
    same way it already finds ties between continuous formulas -- see
    cli.py's "Tied alternatives" section, which no longer needs a
    separate "Tied discrete branches" pass. Completeness isn't
    guaranteed either way -- the spec never required finding *every*
    tied solution (there can be infinitely many in the continuous case,
    e.g. sliding between more Hetonite Part vs. more Yazhen A+C), and a
    perturbation direction finding one specific discrete alternative
    (vs. a different one) is inherently a bit search-order-dependent.
    What matters is surfacing genuinely different solutions a player
    would care about, which this does; whether any *particular* tied
    integer assignment shows up in a short truncated list is not
    something this tries to guarantee.
    """
    formulas = build_formulas(config)
    names = list(formulas.keys())
    idx = {name: i for i, name in enumerate(names)}
    supply = full_supply(config)
    result = maximize_dollar(supply, list(formulas.values()))

    z = int(round(result.formula_rates[idx["xiranite_forge_alloc"]]))
    metatransfer = np.zeros(len(RESOURCE_NAMES))
    for i, mt in enumerate(config.metatransfers):
        opt_name = f"metatransfer_option_{i}"
        metatransfer += result.formula_rates[idx[opt_name]] * np.asarray(
            mt, dtype=float
        )

    return SearchResult(
        result=result,
        z=z,
        metatransfer=metatransfer,
        formula_names=names,
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
