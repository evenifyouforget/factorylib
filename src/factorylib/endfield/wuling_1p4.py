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
Ore (420), Inergen (>=460), and Xiragen (100) are the ONLY materials
with any base income at all, besides the virtual forge_budget
allowance. Every other new material must be crafted -- see
WulingConfig1p4's docstring for how Carbon/Stabilized Carbon's real
supply chain is modeled, and the Fluid-Gas/Solid-Gas Transmuting Unit
note below for how Pyrrolite's own chain closes.

No folded formulas (confirmed with the user): 1.2e collapses Yazhen and
Jincao's Planting+Shredding+Reactor Crucible chain into one zero-cost
step each, since it's otherwise all-Water and there was no scarce
resource whose flow that would hide. That's no longer true once the
Carbon supply chain needs raw Yazhen/Jincao as an *alternative* use for
the same material -- so both are unfolded here into their real 3
stages (yazhen_plant/yazhen_powder_make/yazhen_solution_make and the
Jincao equivalents), verified to still reproduce 1.2e's historical
$-optimal figures exactly (see test_wuling_1p4.py).

Both directions of Fluid-Gas/Solid-Gas Transmuting Unit recipes
(confirmed with the user): kaneko_1p4_data_sheet.md's generic
"(X units of the gas) -> (Y units of the solid/liquid), every Z
seconds (reverse of the above N recipes)" line means the same building
runs the same conversion backwards, at the same ratio and activation
cost, just with input/output swapped -- not a separately-unconfirmed
rate. Confirmed detail: the activation good (Liquid Xiranite for
Fluid-Gas, Xiragen for Solid-Gas) stays on the *input* side even in
reverse -- it's never itself produced by the reverse recipe. This
resolved Pyrrolite's earlier "no confirmed recipe produces it"
gap: Gas Reactor Globe already makes Pyrrolite Gas, and the reverse of
Solid-Gas Transmuting Unit's own Pyrrolite recipe turns that back into
solid Pyrrolite. Two of these 12 reverse formulas
(fluid_gas_aquagen_reverse/fluid_gas_acridgen_reverse) are pure sinks
(their real product, Water/Acid, isn't a tracked resource) -- modeled
anyway for fidelity, though no rational plan would ever run them.

Note this also means an exact 1.2e $-optimal reproduction test can no
longer just zero out Inergen/Xiragen base supply and expect equality
(see test_wuling_1p4.py): the wider gas network can bootstrap real new
value even from zero external gas supply (e.g. converting otherwise-
idle Xiranite into Xiragen via solid_gas_xiragen, itself self-
referential -- Xiragen is both the activation input and the output).
This is expected, not a bug -- adding feasible options to an LP can
only weakly improve its optimum, never hurt it -- so the historical
reproduction tests check ">=" on the full model, plus a separate
"==" check with every purely-additive 1.4 formula explicitly banned
(computed as a set difference against 1.2e's own formula names, not
hand-maintained) to isolate and confirm the unfolding itself is still
exactly lossless.

Known gaps / assumptions (flag before treating any of this as final):
  - Filling Unit is only modeled for the 2 gas variants new_goals.md and
    the exploration-items list actually reference (Inergen, Xiragen),
    not all 8 possible gases the recipe is generically described for.
  - Every "[threshold 6/min]" recipe (kaneko_1p4_data_sheet.md tags this
    on Fluid-Gas/Solid-Gas Transmuting Unit and Gas Dispersing Unit only
    -- NOT Gas Reactor Globe or Purification Unit, whose Xiragen/Cuprium
    Gas inputs are ordinary stoichiometric reactants despite superficially
    resembling one) now models the literal "fixed N/min overhead
    regardless of building utilization" mechanic the science report
    describes (kaneko_1p4_data_sheet.md's "6/min threshold activations...
    aren't actually flow based"), not just a proportional approximation
    of it -- see _THRESHOLD_RECIPES/_threshold_formulas: each recipe
    becomes an integer "{name}_alloc" (one per committed building, paying
    the fixed 6/min cost no matter how much of that building's own
    throughput cap gets used) plus a continuous "{name}_run" (capacity-
    gated, 1-for-1 with real production). Confirmed empirically this is
    NOT always equivalent to the old proportional folding, contrary to an
    earlier draft's assumption: whenever total demand isn't a clean
    multiple of a recipe's per-building cap (30, or 6 for the two "every
    10 seconds" Heavy Xiragen batch recipes), the marginal partially-
    utilized building still pays its full fixed threshold cost under the
    literal model, which the old proportional folding never charged --
    e.g. the default config's $-optimal output is $2171.10 with the
    literal integer model vs. $2183.95 with the old proportional folding
    (fluid_gas_xiragen needs 6 integer buildings for 180 capacity but
    only uses 168, wasting some of the 6th building's fixed Liquid
    Xiranite budget) -- a real ~0.6% loss, not noise.
    WulingConfig1p4.continuous_thresholds=True restores the old
    proportional-folding formulas (single continuous formula per recipe,
    no alloc/capacity split) for before/after comparison; the literal
    integer model is the default, per the user's explicit preference.
    Gas Dispersing Unit is unaffected either way -- always integer,
    never subject to this toggle.
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
    # Two-layer capacity pools for every "[threshold 6/min]" Fluid-Gas/
    # Solid-Gas Transmuting Unit recipe (see _THRESHOLD_RECIPES) -- one
    # per recipe direction, each building's own real per-minute
    # throughput cap (30, or 6 for the "every 10 seconds" Heavy Xiragen
    # batch recipes), consumed 1-for-1 by that recipe's "_run" formula.
    # Only ever populated when WulingConfig1p4.continuous_thresholds is
    # False (the default) -- see build_formulas.
    "fluid_gas_aquagen_capacity",
    "fluid_gas_aquagen_reverse_capacity",
    "fluid_gas_xiragen_capacity",
    "fluid_gas_xiragen_reverse_capacity",
    "fluid_gas_cuprium_gas_capacity",
    "fluid_gas_cuprium_gas_reverse_capacity",
    "fluid_gas_acridgen_capacity",
    "fluid_gas_acridgen_reverse_capacity",
    "fluid_gas_heavy_xiragen_capacity",
    "fluid_gas_heavy_xiragen_reverse_capacity",
    "fluid_gas_hetonite_gas_capacity",
    "fluid_gas_hetonite_gas_reverse_capacity",
    "solid_gas_xiragen_capacity",
    "solid_gas_xiragen_reverse_capacity",
    "solid_gas_heavy_xiragen_capacity",
    "solid_gas_heavy_xiragen_reverse_capacity",
    "solid_gas_cuprium_gas_capacity",
    "solid_gas_cuprium_gas_reverse_capacity",
    "solid_gas_hetonite_gas_capacity",
    "solid_gas_hetonite_gas_reverse_capacity",
    "solid_gas_pyrrolite_gas_capacity",
    "solid_gas_pyrrolite_gas_reverse_capacity",
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
    "fluid_gas_aquagen_capacity": "Fluid-Gas Transmuting Unit (Aquagen) Capacity",
    "fluid_gas_aquagen_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Aquagen, reverse) Capacity"
    ),
    "fluid_gas_xiragen_capacity": "Fluid-Gas Transmuting Unit (Xiragen) Capacity",
    "fluid_gas_xiragen_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Xiragen, reverse) Capacity"
    ),
    "fluid_gas_cuprium_gas_capacity": (
        "Fluid-Gas Transmuting Unit (Cuprium Gas) Capacity"
    ),
    "fluid_gas_cuprium_gas_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Cuprium Gas, reverse) Capacity"
    ),
    "fluid_gas_acridgen_capacity": "Fluid-Gas Transmuting Unit (Acridgen) Capacity",
    "fluid_gas_acridgen_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Acridgen, reverse) Capacity"
    ),
    "fluid_gas_heavy_xiragen_capacity": (
        "Fluid-Gas Transmuting Unit (Heavy Xiragen) Capacity"
    ),
    "fluid_gas_heavy_xiragen_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Heavy Xiragen, reverse) Capacity"
    ),
    "fluid_gas_hetonite_gas_capacity": (
        "Fluid-Gas Transmuting Unit (Hetonite Gas) Capacity"
    ),
    "fluid_gas_hetonite_gas_reverse_capacity": (
        "Fluid-Gas Transmuting Unit (Hetonite Gas, reverse) Capacity"
    ),
    "solid_gas_xiragen_capacity": "Solid-Gas Transmuting Unit (Xiragen) Capacity",
    "solid_gas_xiragen_reverse_capacity": (
        "Solid-Gas Transmuting Unit (Xiragen, reverse) Capacity"
    ),
    "solid_gas_heavy_xiragen_capacity": (
        "Solid-Gas Transmuting Unit (Heavy Xiragen) Capacity"
    ),
    "solid_gas_heavy_xiragen_reverse_capacity": (
        "Solid-Gas Transmuting Unit (Heavy Xiragen, reverse) Capacity"
    ),
    "solid_gas_cuprium_gas_capacity": (
        "Solid-Gas Transmuting Unit (Cuprium Gas) Capacity"
    ),
    "solid_gas_cuprium_gas_reverse_capacity": (
        "Solid-Gas Transmuting Unit (Cuprium Gas, reverse) Capacity"
    ),
    "solid_gas_hetonite_gas_capacity": (
        "Solid-Gas Transmuting Unit (Hetonite Gas) Capacity"
    ),
    "solid_gas_hetonite_gas_reverse_capacity": (
        "Solid-Gas Transmuting Unit (Hetonite Gas, reverse) Capacity"
    ),
    "solid_gas_pyrrolite_gas_capacity": (
        "Solid-Gas Transmuting Unit (Pyrrolite Gas) Capacity"
    ),
    "solid_gas_pyrrolite_gas_reverse_capacity": (
        "Solid-Gas Transmuting Unit (Pyrrolite Gas, reverse) Capacity"
    ),
}


FORMULA_LABELS = {
    # Every 1.2e label is reused as a FALLBACK (safety net for any
    # formula name not explicitly overridden below -- e.g. every "_sell"
    # formula, which stays fine as "X (sellable)": no hidden building or
    # quantity to clarify there), except the two Forge of the Sky
    # allocation formulas 1.4 replaces outright (see build_formulas).
    # Every explicit override below spells out the REAL building name
    # and each item's real /min rate (confirmed with the user: labels
    # like "Cuprium Ore Refining (Cuprium Ore → Cuprium + Sewage)" or
    # "Yazhen Planting (→ Yazhen)" hid both which building runs the
    # recipe and how much of anything it actually needs -- format is
    # "{Building}: {qty}/min {item} [+ ...] → {qty}/min {item} [+ ...]",
    # matching every real recipe ratio exactly as coded in build_formulas
    # (verified against each formula's own consumption vector, not
    # retyped from memory) plus FORMULA_WATTS' implicit Water/Acid
    # quantities where relevant.
    **{
        name: label
        for name, label in v1p2e.FORMULA_LABELS.items()
        if name not in ("xiranite_forge_alloc", "heavy_xiranite_forge_alloc")
    },
    "cup_conv": (
        "Refining Unit: 30/min Cuprium Ore + 30/min Water → "
        "30/min Cuprium + 30/min Sewage"
    ),
    "originium_powder_make": (
        "Shredding Unit: 30/min Originium Ore → 30/min Originium Powder"
    ),
    "ori_to_dop": (
        "Grinding Unit: 60/min Originium Powder + 30/min Sandleaf Powder → "
        "30/min Dense Originium Powder"
    ),
    "ferrium_make": "Refining Unit: 30/min Ferrium Ore → 30/min Ferrium",
    "ferrium_powder_make": "Shredding Unit: 30/min Ferrium → 30/min Ferrium Powder",
    "liquid_xiranite_make": (
        "Reactor Crucible: 30/min Xiranite + 30/min Water → 30/min Liquid Xiranite"
    ),
    "xi_sew": (
        "Reactor Crucible: 30/min Liquid Xiranite + 30/min Sewage → "
        "30/min Xircon Effluent + 30/min Inert Xircon Effluent"
    ),
    "xircon_make": (
        "Reactor Crucible: 60/min Xircon Effluent + 30/min Ferrium Powder → "
        "30/min Xircon + 30/min Sewage"
    ),
    "sc_make": (
        "Packaging Unit: 30/min Xircon + 120/min Dense Originium Powder → "
        "6/min SC Wuling Battery"
    ),
    "lc_make": (
        "Packaging Unit: 30/min Xiranite + 90/min Dense Originium Powder → "
        "6/min LC Wuling Battery"
    ),
    "cuprium_powder_make": "Shredding Unit: 30/min Cuprium → 30/min Cuprium Powder",
    "cuprium_solution_make": (
        "Reactor Crucible: 30/min Cuprium Powder + 30/min Acid → "
        "30/min Cuprium Solution"
    ),
    "hetonite_solution_make": (
        "Purification Unit: 120/min Cuprium Solution → 30/min Hetonite Solution"
    ),
    "hetonite_make": (
        "Reactor Crucible: 60/min Hetonite Solution + 30/min Ferrium Powder → "
        "30/min Hetonite + 30/min Sewage"
    ),
    "hp_make": "Fitting Unit: 30/min Hetonite → 6/min Hetonite Part",
    "hx_make": (
        "Forge of the Sky: 60/min Xiranite + 30/min Xircon Effluent → "
        "6/min Heavy Xiranite"
    ),
    "ferrium_part_make": "Fitting Unit: 30/min Ferrium → 30/min Ferrium Part",
    "cuprium_bottle_make": "Moulding Unit: 60/min Cuprium → 30/min Cuprium Bottle",
    "ferrium_bottle_make": "Moulding Unit: 60/min Ferrium → 30/min Ferrium Bottle",
    "cuprium_part_make": "Fitting Unit: 30/min Cuprium → 30/min Cuprium Part",
    "purify": (
        "Purification Building: 120/min Inert Xircon Effluent → 30/min Xircon Effluent"
    ),
    "purify_node": (
        "Test Area Purification Node: 30/min Sewage → 1/min Xircon Effluent "
        "(max 12 multiples)"
    ),
    "ferrium_component": (
        "Gearing Unit: 60/min Origocrust + 60/min Ferrium → 6/min Ferrium Component"
    ),
    "origocrust_make": "Refining Unit: 30/min Originium Ore → 30/min Origocrust",
    "xiranite_component": (
        "Gearing Unit: 60/min Packed Origocrust + 60/min Xiranite → "
        "6/min Xiranite Component"
    ),
    "packed_origocrust_make": (
        "Grinding Unit: 60/min Origocrust + 30/min Sandleaf Powder → "
        "30/min Packed Origocrust"
    ),
    "cuprium_component": (
        "Gearing Unit: 60/min Cuprium Part + 60/min Xiranite → 6/min Cuprium Component"
    ),
    "hetonite_component": (
        "Gearing Unit: 12/min Hetonite Part + 12/min Heavy Xiranite → "
        "6/min Hetonite Component"
    ),
    # sandleaf_plant is reused unchanged from 1.2e (v1p2e.FORMULA_LABELS'
    # own "Sandleaf Planting (→ Sandleaf)" stays untouched there) -- this
    # overrides ONLY the 1.4 copy, since 1.4 is the one that actually
    # taxes it via FORMULA_WATTS (same real recipe/cost as
    # buckflower_plant: "150 W -> 30 Sandleaf", power_consumption.md).
    "sandleaf_plant": "Planting Unit: 150 W → 30/min Sandleaf",
    "sandleaf_powder": "Shredding Unit: 30/min Sandleaf → 90/min Sandleaf Powder",
    "thermal_bank": "Thermal Bank: 7.5/min Originium Ore → 50 W",
    "dense_ferrium_powder_make": (
        "Grinding Unit: 60/min Ferrium Powder + 30/min Sandleaf Powder → "
        "30/min Dense Ferrium Powder"
    ),
    "steel_make": "Refining Unit: 30/min Dense Ferrium Powder → 30/min Steel",
    "steel_part_make": "Fitting Unit: 30/min Steel → 30/min Steel Part",
    "sc_valley": (
        "Packaging Unit: 60/min Ferrium Part + 90/min Originium Powder → "
        "6/min SC Valley Battery"
    ),
    "hc_valley": (
        "Packaging Unit: 60/min Steel Part + 90/min Dense Originium Powder → "
        "6/min HC Valley Battery"
    ),
    "thermal_bank_sc_valley": "Thermal Bank: 1.5/min SC Valley Battery → 420 W",
    "thermal_bank_hc_valley": "Thermal Bank: 1.5/min HC Valley Battery → 1100 W",
    "sc_power": "Thermal Bank: 1.5/min SC Wuling Battery → 3200 W",
    "lc_power": "Thermal Bank: 1.5/min LC Wuling Battery → 1600 W",
    # ---- Yazhen/Jincao unfolded stages ----
    "yazhen_plant": "Planting Unit: 100 W + 30/min Water → 60/min Yazhen",
    "yazhen_powder_make": "Shredding Unit: 30/min Yazhen → 60/min Yazhen Powder",
    "yazhen_solution_make": (
        "Reactor Crucible: 30/min Yazhen Powder + 30/min Water → 30/min Yazhen Solution"
    ),
    "jincao_plant": "Planting Unit: 100 W + 30/min Water → 60/min Jincao",
    "jincao_powder_make": "Shredding Unit: 30/min Jincao → 60/min Jincao Powder",
    "jincao_solution_make": (
        "Reactor Crucible: 30/min Jincao Powder + 30/min Water → 30/min Jincao Solution"
    ),
    # ---- Carbon / Stabilized Carbon supply chain ----
    "buckflower_plant": "Planting Unit: 150 W → 30/min Buckflower",
    "carbon_from_buckflower": "Refining Unit: 30/min Buckflower → 30/min Carbon",
    "carbon_from_sandleaf": "Refining Unit: 30/min Sandleaf → 30/min Carbon",
    "carbon_from_jincao": "Refining Unit: 30/min Jincao → 60/min Carbon",
    "carbon_from_yazhen": "Refining Unit: 30/min Yazhen → 60/min Carbon",
    "carbon_powder_make": "Shredding Unit: 30/min Carbon → 60/min Carbon Powder",
    "dense_carbon_powder_make": (
        "Grinding Unit: 60/min Carbon Powder + 30/min Sandleaf Powder → "
        "30/min Dense Carbon Powder"
    ),
    "stabilized_carbon_make": (
        "Refining Unit: 30/min Dense Carbon Powder → 30/min Stabilized Carbon"
    ),
    # ---- Forge of the Sky: 3-way integer allocation ----
    "xi_forge_alloc": (
        "Forge of the Sky: 1 building → 30/min Xiranite recipe capacity"
    ),
    "xi_forge_run": (
        "Forge of the Sky: 2/min Stabilized Carbon + 1/min Water → 1/min Xiranite"
    ),
    "xi_forge_stable_env_alloc": (
        "Forge of the Sky, Stable ENV: 1 building → 30/min Xiranite recipe capacity"
    ),
    "xi_forge_stable_env_run": (
        "Forge of the Sky, Stable ENV: 1/min Carbon + 1/min Water → 1/min Xiranite"
    ),
    "hx_forge_alloc": (
        "Forge of the Sky: 1 building → 1/min Heavy Xiranite recipe capacity"
    ),
    # ---- Gas Dispersing Unit ----
    "gas_dispersing_stable": (
        "Gas Dispersing Unit: 6/min Inergen → 4/min Stable Environment Allowance"
    ),
    "gas_dispersing_humid": (
        "Gas Dispersing Unit: 6/min Aquagen → 4/min Humid Environment Allowance"
    ),
    "gas_dispersing_acrid": (
        "Gas Dispersing Unit: 6/min Acridgen → 4/min Acrid Environment Allowance"
    ),
    "gas_dispersing_xiranite_env": (
        "Gas Dispersing Unit: 6/min Xiragen → 4/min Xiranite Environment Allowance"
    ),
    # ---- New continuous recipes ----
    "reactor_crucible_liquid_heavy_xiranite": (
        "Reactor Crucible: 1/min Heavy Xiranite + 1/min Acid → "
        "1/min Liquid Heavy Xiranite"
    ),
    "fitting_unit": "Fitting Unit: 5/min Pyrrolite → 1/min Pyrrolite Part",
    "moulding_unit": (
        "Moulding Unit: 2/min Cuprium + 1/min Inergen → 1/min Cuprium Canister"
    ),
    "gearing_unit": (
        "Gearing Unit: 1/min Pyrrolite + 2/min Heavy Xiranite → "
        "1/min Pyrrolite Component"
    ),
    "filling_unit_inergen": (
        "Filling Unit: 1/min Cuprium Canister + 1/min Inergen → "
        "1/min Cuprium Canister filled with Inergen"
    ),
    "filling_unit_xiragen": (
        "Filling Unit: 1/min Cuprium Canister + 1/min Xiragen → "
        "1/min Cuprium Canister filled with Xiragen"
    ),
    "packaging_unit": (
        "Packaging Unit: 1/min Cuprium Canister + 1/min Xiranite → 2/min Separator Core"
    ),
    "pyrrolite_part_sell": "Pyrrolite Part (sellable)",
    "separator_core_sell": "Separator Core (sellable)",
    "purification_heavy_xiragen": (
        "Purification Unit: 2/min Xiragen + 2/min Separator Core → 1/min Heavy Xiragen"
    ),
    "purification_hetonite_gas": (
        "Purification Unit: 2/min Cuprium Gas + 2/min Separator Core → "
        "1/min Hetonite Gas"
    ),
    # ---- Fluid-Gas Transmuting Unit (both directions) ----
    # Liquid Xiranite's [threshold 6/min] activation is charged via the
    # paired "_alloc" formula (see _threshold_formulas), not this "_run"
    # ratio itself, but is described here as part of the recipe's own
    # real quantities either way -- a reader shouldn't need to know
    # about this module's own internal alloc/run split to see the full
    # real recipe.
    "fluid_gas_aquagen": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Water → 1/min Aquagen"
    ),
    "fluid_gas_aquagen_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Aquagen → 1/min Water"
    ),
    "fluid_gas_xiragen": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Liquid Xiranite → 1/min Xiragen"
    ),
    "fluid_gas_xiragen_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Xiragen → 1/min Liquid Xiranite"
    ),
    "fluid_gas_cuprium_gas": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "2/min Cuprium Solution → 1/min Cuprium Gas"
    ),
    "fluid_gas_cuprium_gas_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Cuprium Gas → 2/min Cuprium Solution"
    ),
    "fluid_gas_acridgen": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Acid → 1/min Acridgen"
    ),
    "fluid_gas_acridgen_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Acridgen → 1/min Acid"
    ),
    "fluid_gas_heavy_xiragen": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "2/min Liquid Heavy Xiranite → 5/min Heavy Xiragen"
    ),
    "fluid_gas_heavy_xiragen_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "5/min Heavy Xiragen → 2/min Liquid Heavy Xiranite"
    ),
    "fluid_gas_hetonite_gas": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Hetonite Solution → 1/min Hetonite Gas"
    ),
    "fluid_gas_hetonite_gas_reverse": (
        "Fluid-Gas Transmuting Unit: Liquid Xiranite[threshold 6/min] + "
        "1/min Hetonite Gas → 1/min Hetonite Solution"
    ),
    # ---- Solid-Gas Transmuting Unit (both directions) ----
    "solid_gas_xiragen": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Xiranite → 1/min Xiragen"
    ),
    "solid_gas_xiragen_reverse": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Xiragen → 1/min Xiranite"
    ),
    "solid_gas_heavy_xiragen": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "2/min Heavy Xiranite → 5/min Heavy Xiragen"
    ),
    "solid_gas_heavy_xiragen_reverse": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "5/min Heavy Xiragen → 2/min Heavy Xiranite"
    ),
    "solid_gas_cuprium_gas": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "2/min Cuprium → 1/min Cuprium Gas"
    ),
    "solid_gas_cuprium_gas_reverse": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Cuprium Gas → 2/min Cuprium"
    ),
    "solid_gas_hetonite_gas": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Hetonite → 2/min Hetonite Gas"
    ),
    "solid_gas_hetonite_gas_reverse": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "2/min Hetonite Gas → 1/min Hetonite"
    ),
    "solid_gas_pyrrolite_gas": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Pyrrolite → 1/min Pyrrolite Gas"
    ),
    "solid_gas_pyrrolite_gas_reverse": (
        "Solid-Gas Transmuting Unit: Xiragen[threshold 6/min] + "
        "1/min Pyrrolite Gas → 1/min Pyrrolite"
    ),
    # ---- Gas Reactor Globe ----
    "gas_reactor_globe_alloc": "Gas Reactor Globe: 1 building → 30/min recipe capacity",
    "gas_reactor_globe_run": (
        "Gas Reactor Globe: 2/min Hetonite Gas + 1/min Xiragen → 1/min Pyrrolite Gas"
    ),
    # ---- Stable-ENV Purification Unit variants ----
    "purification_heavy_xiragen_stable_alloc": (
        "Purification Unit, Stable ENV: 1 building → "
        "30/min Heavy Xiragen recipe capacity"
    ),
    "purification_heavy_xiragen_stable_run": (
        "Purification Unit, Stable ENV: 2/min Xiragen + 1/min Separator Core → "
        "1/min Heavy Xiragen"
    ),
    "purification_hetonite_gas_stable_alloc": (
        "Purification Unit, Stable ENV: 1 building → "
        "30/min Hetonite Gas recipe capacity"
    ),
    "purification_hetonite_gas_stable_run": (
        "Purification Unit, Stable ENV: 2/min Cuprium Gas + 1/min Separator Core → "
        "1/min Hetonite Gas"
    ),
    # ---- Gear-crafting Point chain (flexible_gear_crafting.md's own
    # conversion ratios -- not a real in-game building, a modeling
    # abstraction over the O(N^2) substitution formulas it replaces, so
    # no building name is invented for these) ----
    "component_to_t1": "50/min Xiranite Component → 1/min T1 Crafting Point",
    "component_to_t2": "50/min Cuprium Component → 1/min T2 Crafting Point",
    "component_to_t3": "50/min Hetonite Component → 1/min T3 Crafting Point",
    "component_to_t4": "50/min Pyrrolite Component → 1/min T4 Crafting Point",
    "t4_to_t3": "1/min T4 Crafting Point → 2/min T3 Crafting Point",
    "t3_to_t2": "1/min T3 Crafting Point → 5/min T2 Crafting Point",
    "t2_to_t1": "1/min T2 Crafting Point → 1/min T1 Crafting Point",
}


def is_forge_or_metatransfer_formula(name: str) -> bool:
    """1.4 equivalent of wuling.is_forge_or_metatransfer_formula: True
    for any pure-bookkeeping "_alloc" formula (xi_forge_alloc/
    xi_forge_stable_env_alloc/hx_forge_alloc, gas_reactor_globe_alloc,
    purification_*_stable_alloc, and every threshold recipe's own
    "{name}_alloc" -- see _THRESHOLD_RECIPES/_threshold_formulas) or a
    metatransfer_option_* formula (reused unchanged from 1.2e, same
    name, same reasoning). Every "_alloc" formula shares the same shape:
    a discrete "which building commits to what" integer choice with no
    real material flow of its own beyond a fixed per-building cost,
    consuming/producing only virtual capacity/budget/allowance
    resources -- suffix matching (rather than a hand-maintained name
    list, this module's earlier draft) means a newly-added threshold
    recipe's _alloc automatically gets excluded from complexity pricing
    too, without needing this function edited in lockstep.
    Does NOT cover the paired "_run" formula (xi_forge_run,
    gas_reactor_globe_run, every "{name}_run") -- those consume real
    tracked resources and should be priced normally."""
    return name.endswith("_alloc") or name.startswith("metatransfer_option_")


# Derived from SC Wuling Battery's own $-vs-W economics (confirmed with
# the user, rather than an arbitrary tiny epsilon): sc_sell earns
# $54/item (54*6 $ per 6-item multiple, see wuling.py), while the SAME
# battery converts to 3200 W per 1.5 items/multiple via sc_power. This
# is the $/W "exchange rate" used below to turn a formula's power draw
# into a real (if modest) $ tax, not just a numerically-arbitrary nudge.
_DOLLAR_PER_WATT = 54.0 / (3200.0 / 1.5)  # = 81/3200 = 0.0253125

# Fluid Pump: 5 W -> 60 Water / 10 W -> 60 Acid (tmp_notes/
# power_consumption.md -- confirmed with the user this is the actual
# real-game cost). Water/Acid are otherwise free/unconstrained in this
# model (every formula needing them just omits a consumption term for
# them at all -- see this module's own Water/Acid convention note), so
# every formula that implicitly needs Water or Acid per its own real
# recipe gets an equivalent Watt cost computed from these ratios
# directly, at its own already-modeled per-multiple rate.
_WATTS_PER_WATER = 5.0 / 60.0
_WATTS_PER_ACID = 10.0 / 60.0

# Total Watts drawn per multiple: Planting Units' own direct draw
# (power_consumption.md) plus every other formula's implicit Water/Acid
# draw (old_prompt.md -- the original pre-1.2e recipe list is the only
# source that states these ratios explicitly), for every formula that's
# otherwise completely free of any power/Water/Acid cost in this model.
# Confirmed with the user this should never actually threaten
# feasibility (a single SC/LC Wuling Battery multiple dwarfs even all of
# these at once) -- see build_formulas' own comment for how this becomes
# a real (not just tie-breaking-tiny) negative Formula.output via
# _DOLLAR_PER_WATT, so the LP has a real, non-arbitrary reason to prefer
# power/Carbon-minimizing recipes over wasteful ones -- resolving what
# would otherwise be a pure LP tie (see cli.py's "Tied alternatives"
# section / this module's own Forge of the Sky Carbon-sourcing
# discussion). This tax's own dollar amount is never actually reported,
# though (see power_dollar_tax_paid/search): since it deliberately covers
# only these formulas, not a complete power-consumption model, its
# absolute $ magnitude has no real-world meaning of its own -- only the
# RELATIVE preference it creates between competing recipes matters to
# the solver. What gets shown to the player is the true, untaxed $/min
# at whatever rates the (tax-guided) solver landed on.
FORMULA_WATTS = {
    "buckflower_plant": 150.0,
    "sandleaf_plant": 150.0,
    "yazhen_plant": 100.0 + 30.0 * _WATTS_PER_WATER,
    "jincao_plant": 100.0 + 30.0 * _WATTS_PER_WATER,
    "cup_conv": 30.0 * _WATTS_PER_WATER,
    "liquid_xiranite_make": 30.0 * _WATTS_PER_WATER,
    "cuprium_solution_make": 30.0 * _WATTS_PER_ACID,
    "yazhen_solution_make": 30.0 * _WATTS_PER_WATER,
    "jincao_solution_make": 30.0 * _WATTS_PER_WATER,
    "xi_forge_run": 1.0 * _WATTS_PER_WATER,
    "xi_forge_stable_env_run": 1.0 * _WATTS_PER_WATER,
    # fluid_gas_aquagen/fluid_gas_acridgen always exist as "_run" (see
    # _threshold_formulas -- the formula-name set no longer depends on
    # WulingConfig1p4.continuous_thresholds).
    "fluid_gas_aquagen_run": 1.0 * _WATTS_PER_WATER,
    "fluid_gas_acridgen_run": 1.0 * _WATTS_PER_ACID,
    "reactor_crucible_liquid_heavy_xiranite": 1.0 * _WATTS_PER_ACID,
}

# $/multiple for every formula that earns real money directly -- every
# 1.2e dollar earner reused unchanged, plus 1.4's two new ones (see
# module docstring for their $/item price). Canonical home for this
# table (pp_goals_1p4.py imports it) purely for consolidation -- unlike
# FORMULA_WATTS, build_formulas doesn't need to touch these at all.
DOLLAR_EARNER_OUTPUTS = {
    "sc_sell": 54 * 6,
    "lc_sell": 25 * 6,
    "hp_sell": 48 * 6,
    "hx_sell": 27 * 6,
    "ya": 22 * 6,
    "yc": 16 * 6,
    "jincao_tea": 22 * 6,
    "jincao_drink": 16 * 6,
    "xi_sell": 1.0,
    "cp_sell": 1.0,
    "pyrrolite_part_sell": 70.0,
    "separator_core_sell": 1.0,
}


# Every 1.2e formula this module reuses unchanged keeps its exact
# GOOD_YIELD/batch size; new 1.4 formulas add their own (all confirmed
# 1-multiple-per-item except packaging_unit, which produces 2 Separator
# Core per multiple per its own recipe). No batch-size convention is
# given for the two new sellable goods (pyrrolite_part_sell/
# separator_core_sell), same reasoning as xi_sell/cp_sell's own 1.0.
GOOD_YIELD = {
    **v1p2e.GOOD_YIELD,
    "pyrrolite_part_sell": 1.0,
    "separator_core_sell": 1.0,
    "fitting_unit": 1.0,
    "moulding_unit": 1.0,
    "gearing_unit": 1.0,
    "filling_unit_inergen": 1.0,
    "filling_unit_xiragen": 1.0,
    "packaging_unit": 2.0,
}

# No new power routes in 1.4 (kaneko_1p4_data_sheet.md doesn't mention
# any) -- reused unchanged.
POWER_YIELD = v1p2e.POWER_YIELD

# Priority order goods get sold in once the Wuling Stock Bill cap is hit
# (see wuling.py's own SELL_PRIORITY/factorylib.priority_sell) -- 1.2e's
# order is preserved, with the two new sellable goods appended at the
# end (no data-sheet signal on where they'd rank; append rather than
# guess a specific priority).
SELL_PRIORITY = v1p2e.SELL_PRIORITY + ("pyrrolite_part_sell", "separator_core_sell")

# No new Metatransfer options in 1.4 -- reused unchanged.
METATRANSFER_ITEMS = v1p2e.METATRANSFER_ITEMS

# Zero-$ formulas that exist solely to feed something else (see
# wuling.py's own SECONDARY_GOAL_FORMULA_NAMES/SECONDARY_PLUMBING_FORMULA_NAMES
# docstring) -- reused unchanged from 1.2e. Incomplete for 1.4's own new
# zero-$ intermediates (Cuprium Canister, Separator Core, the Crafting
# Point chain, etc.), which aren't excluded from cli.py's tied-alternatives
# search yet -- this only risks surfacing a few extra, less-interesting
# "tied alternatives" among those, not an incorrect result. Revisit if
# that turns out to be noisy in practice.
SECONDARY_GOAL_FORMULA_NAMES = v1p2e.SECONDARY_GOAL_FORMULA_NAMES
SECONDARY_PLUMBING_FORMULA_NAMES = v1p2e.SECONDARY_PLUMBING_FORMULA_NAMES


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
    # Real, physical, belt-transportable items, same as any other named
    # Component/Part -- unlike 1.2e's own xiranite_component/cuprium_
    # component/hetonite_component (which have no tracked resource of
    # their own at all, see _NEW_RESOURCE_NAMES' comment), these have a
    # real resource dimension in 1.4 (feeding the Crafting Point chain),
    # so they need a belt speed to be correctly treated as delivery-quota
    # candidates and depot-accumulation candidates -- omitting this was a
    # real gap (they'd have silently behaved like bookkeeping dimensions,
    # e.g. invisible to pp_goals_1p4's delivery-quota candidates and
    # cli.py/delivery.py's accumulation-rate tracking).
    "xiranite_component_item": 30.0,
    "cuprium_component_item": 30.0,
    "hetonite_component_item": 30.0,
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


# Every "[threshold 6/min]" Fluid-Gas/Solid-Gas Transmuting Unit recipe
# (kaneko_1p4_data_sheet.md's "Threshold Activations Science Report":
# each committed building consumes a FIXED 6/min of its threshold good
# regardless of how much of its own real throughput cap it actually
# uses -- Gas Dispersing Unit has the same "[threshold 6/min]" tag but
# is handled separately above, always integer, not subject to
# WulingConfig1p4.continuous_thresholds below). Gas Reactor Globe and
# the Purification Unit recipes are NOT threshold-gated (confirmed:
# no "[threshold]" tag on their inputs in the data sheet) -- their
# existing two-layer alloc/run splits gate on Acrid/Stable ENV
# allowance instead, a different mechanic, untouched here.
#
# Each entry: (name, threshold_good, max_rate, other_consumption).
# threshold_good is the recipe's own activation input (Liquid Xiranite
# for every Fluid-Gas recipe, Xiragen for every Solid-Gas one -- stays
# on the input side even in reverse, confirmed with the user). max_rate
# is the recipe's real per-building throughput cap: 30 for every
# "every 2 seconds" recipe (matching belt/pipe speed), 6 for the "every
# 10 seconds" 2-Heavy-Xiranite(or Heavy Xiragen)/5-Heavy-Xiragen batch
# recipes (fluid_gas_heavy_xiragen*/solid_gas_heavy_xiragen*) -- at
# max_rate, the threshold_good's fixed 6/min cost exactly matches what
# the old proportional-folding formulas charged, so _threshold_formulas'
# continuous=True mode reproduces those original formulas' economics
# exactly (see its own docstring). other_consumption is every OTHER real
# reactant/product, in the same per-max_rate-unit-of-`name`'s-own-rate-
# variable terms the original folded formulas already used (verified by
# hand against every one of those formulas -- this table is a lossless
# refactor of them, not new data).
#
# Only the 11 FORWARD recipes are written out -- kaneko_1p4_data_sheet.md
# confirms each reverse recipe keeps the same threshold_good/max_rate and
# swaps every other input/output, which in this module's "positive =
# consumed, negative = produced" convention is exactly a sign flip of
# every entry in other_consumption (verified by hand against all 11 real
# reverse recipes, including the two self-referential ones where
# threshold_good is ALSO one of the other_consumption keys -- e.g.
# fluid_gas_xiragen's {"liquid_xiranite": 1.0, "xiragen": -1.0} flips to
# fluid_gas_xiragen_reverse's {"liquid_xiranite": -1.0, "xiragen": 1.0}).
# _reverse_threshold_recipe derives the reverse from the forward entry
# instead of hand-duplicating it, removing the risk of a forward/reverse
# transcription mismatch entirely.
_THRESHOLD_RECIPES_FORWARD: list[tuple[str, str, float, dict[str, float]]] = [
    ("fluid_gas_aquagen", "liquid_xiranite", 30.0, {"aquagen": -1.0}),
    (
        "fluid_gas_xiragen",
        "liquid_xiranite",
        30.0,
        {"liquid_xiranite": 1.0, "xiragen": -1.0},
    ),
    (
        "fluid_gas_cuprium_gas",
        "liquid_xiranite",
        30.0,
        {"cuprium_solution": 2.0, "cuprium_gas": -1.0},
    ),
    ("fluid_gas_acridgen", "liquid_xiranite", 30.0, {"acridgen": -1.0}),
    (
        "fluid_gas_heavy_xiragen",
        "liquid_xiranite",
        6.0,
        {"liquid_heavy_xiranite": 2.0, "heavy_xiragen": -5.0},
    ),
    (
        "fluid_gas_hetonite_gas",
        "liquid_xiranite",
        30.0,
        {"hetonite_solution": 1.0, "hetonite_gas": -1.0},
    ),
    ("solid_gas_xiragen", "xiragen", 30.0, {"xi": 1.0, "xiragen": -1.0}),
    (
        "solid_gas_heavy_xiragen",
        "xiragen",
        6.0,
        {"heavy_xiranite": 2.0, "heavy_xiragen": -5.0},
    ),
    ("solid_gas_cuprium_gas", "xiragen", 30.0, {"cup": 2.0, "cuprium_gas": -1.0}),
    (
        "solid_gas_hetonite_gas",
        "xiragen",
        30.0,
        {"hetonite": 1.0, "hetonite_gas": -2.0},
    ),
    (
        "solid_gas_pyrrolite_gas",
        "xiragen",
        30.0,
        {"pyrrolite": 1.0, "pyrrolite_gas": -1.0},
    ),
]


def _reverse_threshold_recipe(
    name: str, threshold_good: str, max_rate: float, other_consumption: dict[str, float]
) -> tuple[str, str, float, dict[str, float]]:
    """The reverse of a forward _THRESHOLD_RECIPES_FORWARD entry -- same
    threshold_good/max_rate (never changes direction), every entry in
    other_consumption sign-flipped (see _THRESHOLD_RECIPES_FORWARD's own
    comment for why that's the correct, verified transformation, even for
    the two self-referential recipes)."""
    return (
        f"{name}_reverse",
        threshold_good,
        max_rate,
        {resource: -amount for resource, amount in other_consumption.items()},
    )


_THRESHOLD_RECIPES: list[tuple[str, str, float, dict[str, float]]] = [
    entry
    for forward in _THRESHOLD_RECIPES_FORWARD
    for entry in (forward, _reverse_threshold_recipe(*forward))
]

# Every one of the recipes above already got its base FORMULA_LABELS
# entry keyed by its bare name; every one always materializes as
# "{name}_alloc"/"{name}_run" (see _threshold_formulas -- the
# formula-name set doesn't depend on WulingConfig1p4.continuous_thresholds),
# so derive both labels from the same bare-name description rather than
# hand-duplicating it, since the recipe itself hasn't changed either way,
# only how its threshold cost is modeled. Names come straight from
# _THRESHOLD_RECIPES (not a separately hand-maintained list) so this
# can't drift out of sync with it.
for _name, *_ in _THRESHOLD_RECIPES:
    FORMULA_LABELS[f"{_name}_run"] = FORMULA_LABELS[_name]
    FORMULA_LABELS[f"{_name}_alloc"] = FORMULA_LABELS[_name] + " -- building commitment"
del _name


def _threshold_formulas(
    name: str,
    threshold_good: str,
    max_rate: float,
    other_consumption: dict[str, float],
    continuous: bool,
) -> dict[str, Formula]:
    """Build one `[threshold 6/min]`-gated recipe as the two-layer
    `{name}_alloc`/`{name}_run` pair -- ALWAYS, regardless of
    `continuous`, so the formula-name set this module produces never
    depends on WulingConfig1p4.continuous_thresholds (earlier drafts
    varied the name set itself -- a bare `{name}` under continuous=True
    vs. `{name}_alloc`/`{name}_run` otherwise -- which forced every
    caller needing one of these formulas by name to guess which variant
    existed; see FORMULA_WATTS' git history for three separate bugs that
    caused).

    `{name}_alloc` commits one building at a time, paying the FIXED
    6/min threshold cost regardless of utilization and minting
    `max_rate` units of `{name}_capacity` -- that building's own real
    per-minute throughput cap. `{name}_run` is continuous, consuming
    that capacity 1-for-1 with real throughput alongside every other
    real reactant/product in other_consumption.

    continuous controls only whether `{name}_alloc` is integer=True (the
    default, continuous=False -- a real building count, so a partially-
    utilized building still pays its full fixed threshold cost, the
    whole point of modeling it this way instead of folding the cost in
    proportionally) or integer=False (continuous=True -- a "fractional
    buildings" relaxation, mathematically exact reproduction of the old
    proportional-folding approximation: with alloc unconstrained and
    continuous, the LP can always set alloc's rate to exactly
    run_rate/max_rate, giving the exact same threshold_good-per-output
    ratio as the old single-formula version, no rounding/discretization
    loss -- see module docstring for why this can still differ
    economically from the integer model once utilization isn't free to
    optimize continuously)."""
    capacity = f"{name}_capacity"
    run_vec = dict(other_consumption)
    run_vec[capacity] = run_vec.get(capacity, 0.0) + 1.0
    return {
        f"{name}_alloc": make_formula(
            {threshold_good: 6.0, capacity: -max_rate}, 0.0, integer=not continuous
        ),
        f"{name}_run": make_formula(run_vec, 0.0),
    }


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
# the original pre-1.2e problem statement) is modeled below in
# build_formulas: Buckflower or Sandleaf -> Carbon -> Carbon Powder ->
# Dense Carbon Powder -> Stabilized Carbon. Pyrrolite's own chain closes
# via Gas Reactor Globe (-> Pyrrolite Gas) + the reverse of Solid-Gas
# Transmuting Unit's Pyrrolite recipe (-> solid Pyrrolite) -- see module
# docstring's Fluid-Gas/Solid-Gas Transmuting Unit note.
DEFAULT_ORIGINIUM_ORE = 540.0
DEFAULT_FERRIUM_ORE = 120.0
DEFAULT_CUPRIUM_ORE = 420.0
DEFAULT_INERGEN = 460.0  # "at least 460/min" per the data sheet (was 260)
DEFAULT_XIRAGEN = 100.0  # was 30, per the data sheet's revised income figures
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
        continuous_thresholds: if False (the default), every
            "[threshold 6/min]" Fluid-Gas/Solid-Gas Transmuting Unit
            recipe is modeled as the literal two-layer integer model
            (confirmed with the user: a partially-utilized building
            still pays its full fixed threshold cost, so this should be
            the default, not an opt-in) -- see _THRESHOLD_RECIPES/
            _threshold_formulas. Set True to fall back to the old
            proportional-folding approximation instead, for an
            apples-to-apples before/after comparison. Gas Dispersing
            Unit is unaffected either way -- it's always integer (its
            own "[threshold 6/min]" tag was never modeled any other
            way).
        power_dollar_tax: if True (the default), every formula in
            FORMULA_WATTS gets a real (if modest) negative Formula.output
            (see build_formulas' own comment) purely to guide the LP's
            OWN choice among otherwise-tied vertices -- confirmed with
            the user this resolves what would otherwise be a pure LP tie
            between power/Carbon-wasteful and -efficient recipe choices
            (e.g. Forge of the Sky's Carbon-sourcing split). The reported
            dollar_output is always the real, untaxed figure regardless
            (search() adds the tax back via power_dollar_tax_paid before
            returning) -- the tax's own $ amount has no real-world
            meaning to show a player, only relative value to the solver.
            Set False to disable the tax entirely (every formula keeps
            its plain $0 output) -- needed by the historical
            1.2e-reproduction tests, whose exact `==` dollar assertions
            predate this tax and would otherwise need re-deriving for a
            concern (lossless unfolding) this tax has nothing to do with.
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
    continuous_thresholds: bool = False
    power_dollar_tax: bool = True
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
            # Truncated the same way as base_supply above: metatransfers
            # are length-len(RESOURCE_NAMES) (this module's extended
            # vector, matching this class's own __post_init__ validation
            # and the CLI's --metatransfer help text) but the underlying
            # 1.2e build_formulas() call only knows about its own,
            # shorter RESOURCE_NAMES -- passing the untruncated vector
            # through crashed with a shape mismatch inside
            # v1p2e.build_formulas (caught by
            # test_main_explicit_purify_building_and_metatransfer_flags).
            metatransfers=[
                mt[: len(v1p2e.RESOURCE_NAMES)] for mt in self.metatransfers
            ],
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

    # Fluid-Gas/Solid-Gas Transmuting Unit: both directions modeled for
    # all 6+5 recipes (confirmed with the user -- kaneko_1p4_data_sheet.md's
    # generic "(X units of the gas) -> (Y units of the solid), every Z
    # seconds (reverse of the above N recipes)" line means the same
    # building runs the same conversion backwards, at the same ratio and
    # activation cost, just with input/output swapped -- not a
    # separately-unconfirmed rate). Each recipe's own threshold
    # activation cost (Liquid Xiranite/Xiragen) is modeled per
    # config.continuous_thresholds -- see _THRESHOLD_RECIPES/
    # _threshold_formulas' own docstring for the two-layer-integer-vs-
    # proportional-folding tradeoff this toggles.
    for _tname, _threshold_good, _max_rate, _other in _THRESHOLD_RECIPES:
        f.update(
            _threshold_formulas(
                _tname,
                _threshold_good,
                _max_rate,
                _other,
                continuous=config.continuous_thresholds,
            )
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

    # Virtual power/Water/Acid $ tax (see FORMULA_WATTS' own comment for
    # the full derivation): every formula that draws power (or implicitly
    # needs Water/Acid) gets a small NEGATIVE Formula.output, converted
    # from its own Watt draw via _DOLLAR_PER_WATT. Formula/maximize_dollar
    # only validate `output >= 0` at CONSTRUCTION time (__post_init__) --
    # direct attribute assignment afterward, exactly like the
    # formula_limits/formula_outputs overrides just below already do for
    # .limit/.output, bypasses that check entirely, and scipy's
    # linprog/milp handle a negative objective coefficient correctly
    # either way (it's just a real cost the LP will minimize exposure to,
    # not a special case). Applied BEFORE config.formula_outputs so an
    # explicit override there still wins outright for any of these
    # formulas, same "last write wins" semantics as every other override.
    if config.power_dollar_tax:
        for name, watts in FORMULA_WATTS.items():
            f[name].output = -(watts * _DOLLAR_PER_WATT)

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


def power_dollar_tax_paid(rates_by_name: dict[str, float]) -> float:
    """Total $ the virtual power/Water/Acid tax (FORMULA_WATTS/
    build_formulas' own comment) subtracted from the raw $-maximizing
    objective, at the given rates -- add this back to a raw dollar_output
    to recover the real, untaxed $/min actually shown to the player.
    Confirmed with the user: the tax's own absolute magnitude has no
    real-world meaning of its own (it deliberately covers only some
    buildings, not a complete power-consumption model) -- only relevant
    as a signal for the LP's own vertex choice between otherwise-tied
    configurations, never as something to report. search() already
    applies this to its own result; callers computing $ from formula
    rates independently (e.g. cli.py's tied-alternatives search) need to
    apply it themselves too."""
    return sum(
        rates_by_name.get(name, 0.0) * watts * _DOLLAR_PER_WATT
        for name, watts in FORMULA_WATTS.items()
    )


def search(config: WulingConfig1p4) -> tuple[OptimizeResult, list[str]]:
    """Find the $-optimal 1.4 production plan. Returns (result,
    formula_names) since there's no metatransfer/z bookkeeping to
    surface specially (unlike wuling.SearchResult) -- every discrete
    choice here (forge allocation, environment allocation) is an
    ordinary named formula rate.

    dollar_output is always the real, untaxed $/min (see
    power_dollar_tax_paid) -- config.power_dollar_tax only ever affects
    *which* vertex the LP settles on, never what gets reported for it."""
    formulas = build_formulas(config)
    names = list(formulas.keys())
    supply = full_supply(config)
    result = maximize_dollar(supply, list(formulas.values()))
    if config.power_dollar_tax and result.status == "optimal":
        rates_by_name = dict(zip(names, result.formula_rates))
        result = OptimizeResult(
            status=result.status,
            dollar_output=result.dollar_output + power_dollar_tax_paid(rates_by_name),
            formula_rates=result.formula_rates,
            resource_slack=result.resource_slack,
        )
    return result, names
