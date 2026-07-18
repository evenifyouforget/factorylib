# WIP / TODO notes (factorylib.endfield)

Scratch tracking file for work-in-progress across sessions. Not meant to
ship in a release PR long-term (delete once superseded by real issue
tracking), but committed for now per explicit instruction so it survives
between sessions.

## Where things stand

- PR #13 ("Specialize for Endfield: Endfield-specific multi-goal
  solving, integer formulas, CLI") merged to `main` at `fe099a3`.
- Current branch: `wuling-1p4`, for Endfield 1.4 content.
- Repo state on this branch: 3127 tests passing, `ruff check`/`ruff
  format --check` clean.
- **`src/factorylib/endfield/wuling_1p4.py` now exists** -- the 1.4
  recipe/resource graph, fully unfolded (no folded formulas), both
  directions of every Fluid-Gas/Solid-Gas Transmuting Unit recipe
  modeled (closing Pyrrolite's last remaining gap), and verified to
  reproduce 1.2e's historical $-optimal figures exactly once the
  purely-additive gas economy is isolated out.
- **Now fully wired into pp_goals/refine/cli** -- `pp_goals_1p4.py`
  (`PPGoals1p4`/`build_pp_formulas`/`pp_supply`), `refine.py`'s new
  `refine_1p4()`, and `cli.py` (now defaults to 1.4 for both the
  Optimal and Most-fit sections) all exist -- see "1.4 CLI wiring"
  below for the full writeup.
- Default CLI run (`python -m factorylib.endfield`, now 1.4 by
  default):
  - Optimal ($-only): $2171.10/min (199.2% of $1090 stock-bill goal),
    literal two-layer integer threshold model (see "Threshold recipes"
    above) -- $2183.95/min (200.4%) with `--continuous-thresholds`.
  - Most-fit (refined): reaches the $1090 stock-bill target almost
    exactly under default settings; Power/Delivery-quota/Gear-Component
    lines all print correctly for 1.4's resource set (Pyrrolite
    Component included, via its own `gearing_unit` formula).
  - (1.2e's own historical figures, e.g. $1415.99/$1178.63, are no
    longer what `python -m factorylib.endfield` prints by default --
    those remain accurate for `wuling.py`/`pp_goals.py`/`refine.refine()`
    directly, still covered by their own tests.)

## Applied this session (adjust_power_curve.md / clarification_delivery.md)

- **power_hard_cap_ratio: 1.40 -> 1.10** (`PPGoals`, `pp_goals.py`).
  adjust_power_curve.md's physical argument: a battery's energy reserve
  only overcharges in the "drain and fill" regime, and even there loss
  stays under ~1% near realistic demand; real loss mostly comes from
  balancer-fraction approximation error, and DIGE's ~5% (soft cap,
  unchanged) / ~10% (hard cap) convention already covers essentially
  all practical demands -- "there is no need for 40% excess power".
  Verified empirically (not just applied blindly): with the loose 140%
  cap, dollar output was 103.2% of target; with the tight 110% cap,
  108.1% -- strictly better, confirmed deterministic across 10 seeds,
  and power/quota/other-components unaffected either way. Regression
  test: `test_refine_lower_power_hard_cap_frees_more_dollar_headroom`.
  Does NOT fix Hetonite Component (still exactly 0) -- traced that
  specifically (see below), it's a separate, deeper issue that freeing
  more battery capacity doesn't touch.
- **delivery_box_capacity: 14000 -> 12000** (`PPGoals`, `pp_goals.py`)
  per clarification_delivery.md's correction to the real per-job amount.
- **Labeling bug fixed**: `ya`/`yc`/`jincao_tea`/`jincao_drink`
  (Yazhen Syringe A/C, Jincao Tea/Drink) are dollar-earning formulas
  exactly like `sc_sell`/`hp_sell`/etc., and can equally accumulate
  unsold surplus in the depot -- but unlike those, their
  `FORMULA_LABELS` entries never got the `"(sold)"` suffix. Found via a
  test failure after the power-curve retune shifted which good ends up
  "the" unsold one in `test_main_delivery_prediction_includes_unsold_goods`'s
  forced-low-stock-bill-cap scenario. Now consistent.

### Hetonite Component: sharper root-cause diagnosis (still unresolved)
Traced *why* it's stuck at 0, beyond "resource tension": the real
recipe needs 12 Hetonite Part + 12 Heavy Xiranite per Hetonite Component
multiple, but the Nonzero Production Goal tiers reward reaching just
~0.1 units of Hetonite Part -- so the upstream chain (Cuprium Ore ->
... -> Hetonite Part) only ever gets incentivized to produce a tiny
fraction of a unit, nowhere near the 12 needed to unlock even one
Hetonite Component multiple. This is a **batch-size cliff**, not a
resource-competition issue -- confirmed Heavy Xiranite supply itself is
NOT the bottleneck (`hx_make` runs at the same rate as the $-only
baseline). Lowering power's hard cap (freeing more battery capacity)
did nothing for it, as expected once you see it's not battery-capacity
limited at all. Likely fix once we look at it directly: either give
`pp_hetonite_part`'s Nonzero Production Goal tiers a first_cap closer
to the real 12-unit threshold (so partial progress isn't "free" until
the recipe can actually complete), or a dedicated
`hard_satisfaction_bonus`-style gate sized to the real batch
requirement. Not attempted yet -- flagged as the concrete next step if
we want to close this out before 1.4 recipes land.

## Carried-over / deferred scope (from PR #13's own MR checklist)

- **Hetonite Component stuck at 0** -- root-caused further this session
  to a batch-size cliff (see "Hetonite Component: sharper root-cause
  diagnosis" above), not the 3-way resource tension originally
  suspected (confirmed freeing more battery capacity via the power-curve
  fix doesn't move it at all). **Revisit once 1.4's real recipes are
  in** -- new Hetonite routes may resolve this on their own, making
  further tuning of the *old* recipe set moot.
- **Multi-seed "best-of-N" refine wrapper** -- the SA search is
  single-seed by default (`--refine-seed`/`-R` exist for manual
  control, but nothing automatically tries several seeds and keeps the
  best). Mentioned early on, never designed in detail. Given how
  deterministic the search turned out to be across seeds once actual
  bugs were fixed (dozens of seeds landing on bit-identical results),
  this may matter less than originally thought -- reassess before
  building it.
- **Backpressure-corrected "shadow multiples" for liquids** -- explicitly
  deferred as a dedicated follow-up. Joint outputs mean throttling one
  producer can starve a different downstream consumer in ways this
  steady-state LP doesn't model (see factorylib.endfield.goals and
  factorylib.search's module docstrings' repeated caveats about this).
  Real, but a harder fixed-point problem.
- **Modularity requirement for a future SA-replacement PR** (per
  `tmp_notes/future_work.md`): to let one future PR swap out simulated
  annealing for a tuned alternative without touching the rest of the
  pipeline, the search's own candidate-state representation needs `eq`,
  `hash`, `get_fitness()`, and `get_random_new_mutant()`. **Not true
  today**: `search.py`'s `simulated_annealing()` operates directly on a
  plain `np.ndarray` of formula rates plus a list of `_move` functions
  (`_round_down_move`/`_round_up_move`/`_allocate_slack_move`/
  `_shift_move`/`_toggle_integer_move`/`_pinned_lp_move`) and a fitness
  callback passed in from the caller (`refine.py`) -- there's no
  dedicated state class at all, so eq/hash/mutate aren't meaningful yet.
  Confirmed: no direction/gradient info is tracked (rules out gradient
  descent) and there's no crossbreeding between two candidate states
  (rules out a genetic algorithm) -- but most other metaheuristics
  (tabu search, other annealing schedules, hill-climbing variants) would
  still be possible once a real state class exists. Not attempted yet --
  flagged here so a future PR doing this refactor knows the actual
  current shape of `search.py`, not just the abstract ask.
- **Old scratch files** (`factorylib_tmp_linearization.md`,
  `factorylib_tmp_linearization_notes.md`,
  `factorylib_tmp_physical_factory_construction.md`, the old MR draft)
  -- confirmed already gone from `main` after the merge, nothing to do.

## Endfield 1.4 prep (see kaneko_1p4_data_sheet.md / 1p4_new_features.md)

**Status: recipes/numbers not final yet** ("exact recipes and numbers
are on the way" per 1p4_new_features.md) -- nothing to implement for
real yet, just architecture prep so we're not starting cold once
numbers land.

### New resources / tiers
- Inergen (gas), Xiragen (gaseous Xiranite) -- new environment-sourced
  resources, presumably new RESOURCE_NAMES entries with a "gas" belt
  speed/storage rule analogous to how liquids (belt_speed=120) are
  already excluded from delivery-job/depot accumulation (see
  `factorylib/endfield/delivery.py`'s `_SOLID_BELT_SPEED` filter --
  gases will need the same "can't stash in a depot" treatment as
  liquids, whatever their actual belt speed constant ends up being).
- Pyrrolite: a new tier above Hetonite. Promises cheaper gear crafting
  (see "Alternate gear crafting" below). New Component type
  (`pyrrolite_component`), needs a new own-flow dimension
  (`pyrrolite_component_flow`) alongside the existing
  xiranite/cuprium/hetonite ones in `pp_goals.py`'s
  `_COMPONENT_OWN_FLOWS`, plus a new Nonzero Production Goal tier and
  CLI gear-warning line, mirroring the existing three.

### Threshold / activation formulas -- directly maps to existing infra
1p4_new_features.md's own worked example already describes exactly our
`Formula.integer=True` all-or-nothing pattern:
```
6/min Electrite -> 1 Electrite Activation (integer multiples only)
1 Electrite Activation -> 1 Hydrolysis Electrite Activation (integer multiples only)
30/min Distilled Water + 1 Hydrolysis Electrite Activation -> ...
```
This is structurally identical to `hard_satisfaction_bonus`/
`_materialize_bonus`'s "consume a fixed amount, produce exactly one
indivisible unit of a downstream-gating resource" pattern already in
`pp_goals.py`, and to `wuling.py`'s existing `xiranite_forge_alloc`
/`metatransfer_option_*` "one shared integer-capped pool, multiple
competing consumers" pattern. The "distinct counter per consuming
recipe" requirement (to prevent "sharing a building" between two
half-multiples) is also already exactly what `forge_budget`/
`hx_forge_capacity`/`metatransfer_allowance` do in `wuling.py`. No new
search/optimize machinery needed here -- just new Formula entries once
real recipes exist. The fix from PR #13 (search.py's
`_integer_rates_valid`/`_toggle_integer_move`/`_snap_integer_rates`,
plus `lp_with_floor` preserving `Formula.integer`) means this pattern
is now actually *safe* to use in the SA-refined search too, not just
the raw $-maximizing MILP path -- this wasn't true before that fix.

### Gas Dispersing Unit / environments
`6/min Inergen -> 4 Inergen Environment Allowance (integer multiples
only)` -- same integer-pool pattern as above, one pool per gas type,
consumed by whichever environment-gated recipes are built inside that
field. The "maybe only 3 buildings fit, not 4, if a user doesn't want a
cursed layout" note suggests this should be a configurable constant
(a la `WulingConfig.max_forges`), not hardcoded to 4.

### Flow control (Item/Pipe Control Port, integer N/min limiter, 0..30)
Notable per 1p4_new_features.md: makes 1/5 no longer a "hard" prime
denominator in at least one tutorial-highlighted case (6/min out of
30/min = 1/5) -- this is exactly the kind of thing
`factorylib.simplicity`'s prime-factor complexity pricing already
models (small prime factors are cheap). Once we know which specific
denominators become "free" via this feature (if any beyond the
tutorial's 6/min example), `_NICE_DENOMINATORS`/`fraction_complexity`
may need new cheap-denominator entries -- **but don't guess at this
until real numbers/examples are confirmed**, since the note explicitly
says "a true 1/5 splitter is still hard" in general, just not in this
one case.

**Cross-reference**: `/workspaces/claude-code-devcontainers/batterylib`
(sibling repo) already models the *general* virtual-splitter problem --
`ALL_VSPLITTER_DEFS` in `batterylib/models.py` gives real achievable
balancer fractions with an associated weight_cost, including 1/5 and
1/7 (both via a PASS/RECYCLE/DISCARD cycle, weight_cost 50/60 --
notably *not* free, unlike our current `_NICE_DENOMINATORS` which is
pure powers of 2 and 3 with no cost gradation at all). If the 1.4 flow
limiter turns out to make a genuinely wide set of denominators cheap
(not just the one tutorial example), it may be worth pulling
`ALL_VSPLITTER_DEFS`-style weighted costs into `factorylib.simplicity`
instead of the current flat "in _NICE_DENOMINATORS or not" model --
but only once real 1.4 numbers confirm this is worth the complexity.

### Forge of the Sky: real recipes now confirmed (was previously an abstraction)
Real recipes, replacing 1.2e's abstracted `forge_budget`-allocated
`xiranite_forge_alloc`/`heavy_xiranite_forge_alloc`. **3 total recipes**
a Forge of the Sky building can run (confirmed with the user):
```
Forge of the Sky: 2 Stabilized Carbon + 1 Water -> 1 Xiranite (every 2s)
Forge of the Sky: 1 Carbon + 1 Water -> 1 Xiranite (every 2s, Stable ENV)
Forge of the Sky: 10 Xiranite + 5 Xircon Effluent -> 1 Heavy Xiranite (every 10s)
```
The second recipe (from 1p4_new_features.md/kaneko_1p4_data_sheet.md's
original "New Recipes" listing) is a cheaper Stable-ENV-gated
alternative to the first (1 Carbon instead of 2 Stabilized Carbon) --
so a building placed inside a Gas Dispersing Unit's Stable field has a
strictly better option for Xiranite than one that isn't. Heavy Xiranite
is now downstream of Xiranite (a real consumption, not a separate
capacity allocation) rather than the old two-parallel-outputs framing.

**Confirmed with the user**: global cap is 12 Forge of the Sky
buildings total (same number as 1.2e's `max_forges=12` default -- good
continuity), each committed to exactly ONE of these 3 recipes at a
time -- so the *shared-integer-pool, competing consumers* mechanic from
1.2e (`forge_budget`) is still the right shape, just with 3 competing
allocations instead of 2, and feeding real recipe formulas instead of
an abstraction. This is closer to `wuling.py`'s existing
`metatransfer_option_*` pattern (N mutually-exclusive integer choices
drawing from one shared pool) than the old simple xi/hx 2-way split:
keep a `forge_budget`-like virtual resource, with three
`integer=True` Formula entries (`xi_forge_alloc`,
`xi_forge_stable_env_alloc`, `hx_forge_alloc`) each consuming 1 unit of
it and producing the *capacity* to run the corresponding real recipe.
The Stable-ENV variant additionally needs to be gated on Stable ENV
actually being active (see the Gas Dispersing Unit / environments
section below) -- two independent integer constraints stacked on one
recipe, not just one.

### Dollar target: single combined pool (confirmed)
**Confirmed with the user**: model the new Cloudseeder Station outpost
and the existing Cardiac Remediation Station as ONE combined
`dollar_target`, not two independent ones -- simpler, and avoids
needing to model which specific goods count against which outpost's
cap. Revisit only if this turns out to misprice something once real
sellable-goods overlap/conflicts are checked against both outposts'
lists.

### Alternate gear crafting (Component substitution hierarchy)
Higher-tier Components can substitute for lower-tier gear recipes
(Hetonite Component -> Xiranite-tier gears, but not reverse). Pyrrolite
Component gets a 99% Wuling-Stock-Bill-cost discount, effectively
making it $-free and only Component-supply-limited -- per the notes
author's own read, this may make producing Xiranite/Cuprium/Hetonite
Component "completely unnecessary for endgame users" once Pyrrolite
exists.

**Better design than my first take** (see flexible_gear_crafting.md):
rather than O(N^2) substitution Formula entries (one per eligible
Component-type x gear-tier pair -- 4+3+2+1=10 entries across 4 tiers,
growing quadratically every time a new tier gets added in a future
update), use a linear "Crafting Point" conversion chain instead:
```
50 Pyrrolite Component -> 1 T4 Crafting Point
50 Hetonite Component  -> 1 T3 Crafting Point
50 Cuprium Component   -> 1 T2 Crafting Point
50 Xiranite Component  -> 1 T1 Crafting Point
1 T4 Crafting Point -> 2 T3 Crafting Point
1 T3 Crafting Point -> 5 T2 Crafting Point
1 T2 Crafting Point -> 1 T1 Crafting Point
```
Each gear tier's own crafting formula then just consumes its own
`T_n Crafting Point` (not a direct Component), so producing ONLY
Pyrrolite Component and cascading it down through the chain can supply
all 4 gear tiers -- adding a future T5 tier means adding one new
Component->Point conversion plus one new Point->Point step down, not
retrofitting every existing lower tier. Same "just more ordinary
Formula entries competing in the same LP" principle as my original
idea, just a better *shape* for it. The 99% discount isn't modeled in
this scheme yet -- the working hypothesis is the LP will prefer
Pyrrolite Component anyway once given the choice; only add explicit
discount-modeling if that turns out not to hold once real numbers are
in. Conversion ratios above (50:1, 2/5/1 for T4->T3->T2->T1) are the
current best guess, not yet confirmed against real 1.4 data.

**Update: now confirmed, not just a guess.** kaneko_1p4_data_sheet.md's
"Crafting" section gives the real per-tier $+Component costs (T1: 8k$+
50XC / 8k$+50CC / 8k$+10HC / 80$+5PC; T2: 16k$+50CC / 16k$+10HC / 160$+
5PC; T3: 25k$+50HC / 250$+25PC; T4: 360$+50PC). Working through the
proposed 50:1-own-tier + 2/5/1-cascade ratios by hand reproduces every
single one of these numbers exactly (e.g. 5 PC -> 0.1 T4 CP -> 0.2 T3
CP -> 1.0 T2 CP -> 1.0 T1 CP, matching "5 PC covers T1 or T2" exactly;
10 HC -> 0.2 T3 CP -> 1.0 T2 CP -> 1.0 T1 CP, matching "10 HC covers T1
or T2" too). The Crafting Point design isn't just architecturally
nicer than O(N^2) substitution formulas -- it's mathematically
equivalent to the real costs. Safe to implement as designed.

### "Kaneko's 6/min Science" (speculative, unconfirmed)
Open questions about how flow-rate limiting is actually implemented
under the hood (tick-based cooldown vs. windowed-average leniency), and
whether Control Ports still tank offline efficiency the way they
historically have. Explicitly "yet to be done" experiments by the
community, not something for us to model speculatively -- wait for
confirmed mechanics before encoding anything here.

## 1.4 implementation status (wuling_1p4.py)

**Fork-vs-replace question above is resolved**: went with a clean fork
-- `src/factorylib/endfield/wuling_1p4.py` reuses every unchanged 1.2e
formula (via `wuling.build_formulas()` + zero-padding to the extended
resource vector) and only replaces/adds what 1.4 actually changes. 1.2e
itself is completely untouched. 94 formulas total, 18 tests in
`tests/endfield/test_wuling_1p4.py`, all passing.

**Built and validated**:
- Forge of the Sky: 3-way integer allocation (12 total buildings,
  shared `forge_budget` pool) across the plain Xiranite recipe (2
  Stabilized Carbon), the Stable-ENV Xiranite recipe (1 Carbon,
  confirmed cheaper), and Heavy Xiranite (reuses 1.2e's `hx_make`
  unchanged -- its 60:30:6 ratio exactly matches the newly-confirmed
  10:5:1 recipe scaled by 6, a nice cross-validation). `search()`
  correctly prefers the Stable-ENV route and reproduces the *exact same*
  10/2 forge split 1.2e's own $-optimal baseline finds.
- Gas Dispersing Unit (4 environments, 6/min gas -> 4 allowance each,
  integer).
- All 11 forward-direction new recipes with concrete rates (Fitting/
  Moulding/Gearing/Filling x2/Packaging/Purification x2 base + x2
  Stable-ENV/Gas Reactor Globe/6 Fluid-Gas + 5 Solid-Gas Transmuting).
- Crafting Point chain -- validated by test against every real per-tier
  gear cost in kaneko_1p4_data_sheet.md (not just the earlier by-hand
  check).
- Two new sell formulas (`pyrrolite_part_sell` $70/item,
  `separator_core_sell` $1/item) -- these are given $ prices directly in
  the data sheet, not a tunable goal, so wired in now rather than
  deferred.

**Real bugs caught and fixed while writing this** (worth knowing about
even though they're already fixed):
- Two of the "self-referential" gas conversions (Solid-Gas Transmuting
  Unit's Xiragen->Xiragen recipe, and the Liquid-Heavy-Xiranite->Heavy
  Xiragen recipe) had the threshold-activation input be the *same*
  resource as the recipe's own reactant/product -- an early draft either
  forgot the production term entirely (solid_gas_xiragen) or miscounted
  the batch ratio (fluid_gas_heavy_xiragen, a 5-item-per-cycle recipe
  where per-unit normalization is error-prone). Both caught by hand
  re-deriving from first principles before writing tests, not by the
  tests themselves -- there wasn't an automated check that would have
  caught either, which is itself worth noting as a coverage gap.
- Giving Pyrrolite a `math.inf` placeholder supply made the LP
  genuinely unbounded once `pyrrolite_part_sell` was added (infinite
  Pyrrolite -> infinite $). Switched Carbon/Stabilized Carbon to a
  large-but-finite placeholder (`_PLACEHOLDER_SUPPLY = 10_000.0`)
  instead. Still not a real number, must be replaced once real supply
  data exists.
- Missing the "Reactor Crucible: 1 Heavy Xiranite + 1 Acid -> 1 Liquid
  Heavy Xiranite" formula entirely -- `liquid_heavy_xiranite` had a
  consumer (`fluid_gas_heavy_xiragen`) but no producer and no base
  supply, silently making that recipe permanently infeasible. Caught by
  re-auditing the whole Xiranite/Heavy Xiranite solid/liquid/gas family
  for the "two things wrongly treated as fungible" bug class this
  project has been bitten by before (DOP/Origocrust) -- this was the
  opposite failure mode (a real distinction with no bridge at all, not
  two things merged into one). Fixed; added a general regression test
  (`test_every_new_resource_has_base_supply_or_a_producer`) plus
  targeted ones confirming Xiranite's and Heavy Xiranite's three forms
  stay genuinely distinct and non-free to convert between.

**Confirmed with the user, applied**:
- Ferrium Ore's 120 (vs. 1.2e's own 90) is correct for 1.4; 1.2e's
  default stays at 90 untouched.
- **Pyrrolite has zero base supply** (it must be crafted, not sourced
  directly) -- but no confirmed recipe produces raw Pyrrolite either.
  The only visible path is the *reverse* of "Solid-Gas Transmuting Unit:
  Xiragen[6/min] + 1 Pyrrolite -> 1 Pyrrolite Gas" (turning Pyrrolite
  Gas back into solid Pyrrolite), one of the 6 unmodeled reverse
  Transmuting Unit recipes (rate never given). Until that's confirmed,
  Pyrrolite -- and everything downstream (Pyrrolite Part, Pyrrolite
  Component, the T4 Crafting Point tier, pyrrolite_part_sell) -- is
  structurally unreachable, not just currently at rate 0. Deliberately
  not papered over with a guessed reverse-recipe ratio (e.g. mirroring
  the forward 1:1) since a wrong guess would propagate through
  Pyrrolite's whole downstream chain; a dedicated test
  (`test_pyrrolite_has_no_base_supply_and_is_currently_unreachable`)
  documents this so it isn't silently "fixed" later without deciding to.
  Default config's $-optimal output dropped from $141590 (dominated by
  the since-removed placeholder Pyrrolite supply feeding
  pyrrolite_part_sell) back to a sane $1590.60.

## Carbon/Stabilized Carbon supply chain + full unfolding (old_prompt.md)

The user brought back the original pre-1.2e problem statement
(`tmp_notes/old_prompt.md`), which turned out to have the real recipe
chain feeding Carbon/Stabilized Carbon that no 1.4-specific note
mentioned at all (1.2e never needed it since its Xiranite production
was fully abstracted). Also confirmed: **only 5 materials get any base
income at all** (Originium/Ferrium/Cuprium Ore, Inergen, Xiragen) --
Carbon, Stabilized Carbon, and Pyrrolite must all be crafted. This
initially looked like it would make Xiranite (and therefore almost the
whole economy) unreachable, same as Pyrrolite -- but old_prompt.md
supplied the missing chain:

```
Buckflower or Sandleaf -> Carbon (30:30)
Jincao or Yazhen -> Carbon (30:60, twice as efficient)
Carbon -> Carbon Powder (30:60)
Carbon Powder + Sandleaf Powder -> Dense Carbon Powder (60+30:30)
Dense Carbon Powder -> Stabilized Carbon (30:30)
```

**Confirmed with the user: model all 4 Carbon sources**, including
Jincao/Yazhen -- which required fully unfolding 1.2e's
`yazhen_solution_make`/`jincao_solution_make` (previously one zero-cost
step collapsing Planting+Shredding+Reactor Crucible, since it was
otherwise all-Water) into their real 3 stages, since raw Yazhen/Jincao
is now a genuine competing resource. **Confirmed with the user: no
folded formulas should remain anywhere, and the result must still
match past results** -- now verified by a dedicated regression test
(`test_reproduces_1p2e_historical_dollar_figure_exactly`) that
reproduces 1.2e's exact historical $1415.99... (206735/146) figure
given the same base supply, plus a second one for the ban-ya invariant
(205129/146).

**Two more real bugs found while getting that reproduction test to
actually pass** (the unfolding alone wasn't the whole story):
- `WulingConfig1p4._v1p2e_config()` was passing `metatransfers=[]` to
  the underlying 1.2e config -- silently making `metatransfer_option_0`
  (used at rate 1.0 in 1.2e's own historical $-optimal solution)
  permanently unavailable. Now defaults to `wuling.DEFAULT_METATRANSFERS`.
- `full_supply()` never credited `metatransfer_allowance` at all, so
  enabling metatransfers alone wouldn't have been enough either --
  fixed alongside the above.
- Separately (not a bug, a genuine new constraint): 1.2e's own
  `sandleaf_plant` limit (5, "sized to comfortably cover its tracked
  consumers' floor demand") is too tight once Sandleaf/Sandleaf Powder
  gets a brand-new competing consumer (`dense_carbon_powder_make`).
  Empirically found 10 is the exact minimum needed to still reach
  1.2e's historical figure; raised the default (`DEFAULT_PLANTING_LIMIT`,
  now also shared by the new `buckflower_plant`/`yazhen_plant`/
  `jincao_plant`) to 15 for a small margin, and `build_formulas`
  overrides 1.2e's inherited `sandleaf_plant` to match.

**Not yet done** (deliberately out of scope for this pass, per "most of
the work is in the new recipes"):
- No `pp_goals`/`refine`/`cli` wiring for 1.4 at all yet -- no
  `PPGoals1p4`, no nonzero-goal tiers for the new_goals.md priority list
  (Separator Core/Cuprium Canister variants at 0.1/min, Liquid Heavy
  Xiranite/Cuprium Component/Hetonite Part/Pyrrolite Part/T1-T4 Crafting
  Point at 0.5/min, Liquid Xiranite at 10/min), no CLI entry point.
  `FORMULA_LABELS` now exists (see below).
- Filling Unit only modeled for Inergen/Xiragen (the two variants
  anything downstream actually needs).

### FORMULA_LABELS added; complexity-pricing exclusion generalized
- `wuling_1p4.FORMULA_LABELS` now exists, mirroring `wuling.py`'s own
  (reuses every unchanged 1.2e label, overrides the two replaced Forge
  of the Sky allocations, adds labels for every new formula).
- `goals.py`'s `_plan_complexity` generalized to accept
  `resource_names`/`resource_belt_speed`/`is_bookkeeping_formula` as
  optional parameters (defaulting to 1.2e's exact original behavior,
  verified by `test_goals.py`/`test_refine.py` staying green) --
  prerequisite for reusing it against a 1.4 plan.
- `wuling_1p4.is_forge_or_metatransfer_formula` added: now a generic
  `name.endswith("_alloc")` check (plus the existing
  `metatransfer_option_*` prefix) rather than a hand-maintained name
  list, so every current and future "_alloc" bookkeeping formula
  (Forge of the Sky, Gas Reactor Globe, Purification Stable-ENV
  variants, and now every threshold recipe's own alloc -- see below) is
  automatically excluded from `_plan_complexity`'s belt-fraction
  pricing without needing this function edited in lockstep.

### Threshold recipes: literal two-layer integer model (not just proportional folding)
Per direct instruction: a partially-utilized building still pays its
full fixed 6/min threshold cost in the real game
(kaneko_1p4_data_sheet.md's science report: "6/min threshold
activations aren't actually flow based... after receiving 1 item, the
building is on for 10 seconds") -- an earlier draft's proportional
folding (threshold cost scaled continuously with the recipe's own rate)
was only an approximation, not confirmed equivalent. Confirmed via the
data sheet that **only** Fluid-Gas Transmuting Unit (12 recipes),
Solid-Gas Transmuting Unit (10 recipes), and Gas Dispersing Unit (4,
already integer=True, untouched) carry an actual `[threshold 6/min]`
tag -- Gas Reactor Globe and Purification Unit do NOT (their Xiragen/
Cuprium Gas inputs are ordinary stoichiometric reactants, already
correctly modeled via the existing ENV-gated alloc/run pattern).

**Implemented**: `_THRESHOLD_RECIPES` (a `(name, threshold_good,
max_rate, other_consumption)` table -- `max_rate` is 30 for every
"every 2 seconds" recipe, 6 for the two "every 10 seconds" Heavy
Xiragen batch recipes) + `_threshold_formulas()`, which builds either:
- the literal model (default): `{name}_alloc` (integer, unconstrained
  count, pays the fixed 6/min threshold cost per committed building,
  mints `max_rate` units of `{name}_capacity`) + `{name}_run`
  (continuous, consumes that capacity 1-for-1 with real throughput).
- the old proportional-folding approximation
  (`WulingConfig1p4.continuous_thresholds=True`): a single continuous
  formula, algebraically identical to what this module had before this
  change (verified: the whole 22-formula table was derived by hand
  from the original formulas, not new data).

**Confirmed empirically this is NOT just a modeling nicety -- it changes
the $-optimal answer**: default config's $-optimal output is
**$2171.10** with the literal integer model vs. **$2183.95** with the
old proportional folding (**-0.59%**). Traced the cause: `fluid_gas_xiragen`
needs 6 integer buildings to cover its needed throughput (180 units of
capacity), but only uses 168 of them -- the 6th building's fixed Liquid
Xiranite budget is partially wasted, something the old proportional
model never charged (it always billed exactly 0.2 Liquid Xiranite per
unit output, regardless of "whole building" granularity). This is a real,
structural effect of the literal mechanic, not noise -- confirmed
reproducible, not a solver artifact.

**Still needed**: task #42 (CLI wiring to 1.4) must also add a
`--continuous-thresholds` flag through to `WulingConfig1p4` for
before/after comparison from the command line -- not done yet, since
`cli.py` doesn't wire up `wuling_1p4` at all yet.

## Pyrrolite resolved: both directions of Transmuting Unit recipes

The user pointed out the missing piece directly: Gas Reactor Globe
already makes Pyrrolite Gas (2 Hetonite Gas + 1 Xiragen -> 1 Pyrrolite
Gas, Acrid ENV), and the *reverse* of Solid-Gas Transmuting Unit's own
"Xiragen[6/min] + 1 Pyrrolite -> 1 Pyrrolite Gas" turns that back into
solid Pyrrolite. **Confirmed with the user: both directions of ALL 6
Fluid-Gas and 5 Solid-Gas Transmuting Unit recipes are modeled** (not
just the one Pyrrolite needed) -- the same building runs the same
conversion backwards, same ratio and activation cost, input/output
swapped. Confirmed detail: the activation good (Liquid Xiranite /
Xiragen) always stays on the input side, even in reverse -- it's never
itself produced by a reverse recipe (this matters for the two
self-referential recipes, fluid_gas_xiragen/solid_gas_xiragen, where
the activation good is also the recipe's own reactant/product -- see
module docstring for the exact net-consumption derivation).

**This changed what "reproduces past results" means.** With the wider
gas network in place, the model can bootstrap real new value even from
*zero* external Inergen/Xiragen supply (e.g. converting idle Xiranite
into Xiragen via the self-referential solid_gas_xiragen, then routing
it through the rest of the network) -- so an exact `==` match against
1.2e's historical $-optimal figure no longer holds on the full model.
This is expected, not a regression (adding feasible options to an LP
can only weakly improve its optimum, never hurt it). Split into two
kinds of test accordingly:
- `test_matches_or_exceeds_1p2e_*`: the full model, checking `>=`.
- `test_reproduces_1p2e_*_exactly_with_gas_economy_disabled`: bans
  every purely-additive 1.4 formula (computed as a set difference
  against plain 1.2e's own formula names -- not hand-maintained, so it
  can't go stale as more formulas get added), keeping only the required
  replacement infrastructure (Forge of the Sky's new recipes, the
  Carbon chain feeding them, Yazhen/Jincao's unfolded stages) active.
  Confirmed empirically (not just reasoned about) that this reproduces
  both historical figures (206735/146, 205129/146) bit-exactly.

Also had to relax `test_pyrrolite_...`/`test_search_prefers_stable_env_...`
from "the unconstrained optimum always chooses this path" to "this path
is genuinely feasible under enough incentive" -- once Pyrrolite/Stable
ENV compete with several *other* new reverse recipes for the same
scarce resources (Acrid ENV, Hetonite Gas, Xiragen), which specific
path the unconstrained $-optimal picks becomes a genuine economic
choice each time (like which of two tied alternatives 1.2e's own search
picks), not a fixed guarantee -- feasibility-under-incentive is the
right invariant to test, not "the default config happens to pick it."

## 1.4 CLI wiring: pp_goals_1p4.py, refine_1p4(), cli.py now defaults to 1.4

Confirmed with the user: "Full" scope -- both the Optimal ($-only) and
Most-fit (refined) sections of `python -m factorylib.endfield` now
default to 1.4, not 1.2e. 1.2e itself (`wuling.py`/`pp_goals.py`/
`refine.refine()`) is completely untouched and still fully tested on
its own.

**`src/factorylib/endfield/pp_goals_1p4.py`** (new file): mirrors
`pp_goals.py`'s structure, reusing its generic tier generators
(`satisfaction_tiers`/`nonzero_production_tiers`/`hard_satisfaction_bonus`)
unchanged. `PPGoals1p4` adds new_goals.md's full priority list as
Nonzero Production Goals (Low/Mid/High priority just means a bigger
`first_cap`: 0.1/0.5/10.0) -- Separator Core, both Cuprium Canister
filled variants, Liquid Heavy Xiranite, Pyrrolite Part, and the whole
T1-T4 Crafting Point chain collapsed into ONE goal keyed on
`t1_crafting_point` alone (every tier cascades down to T1 for free, so
rewarding only the bottom of the cascade is sufficient -- matches
new_goals.md's own single-line framing). Raises Cuprium Component's and
Hetonite Part's `first_cap` to 0.5 (Xiranite Component's stays 0.1,
per new_goals.md not listing it). **Resolved the open design question**
from the previous session: Xiranite/Cuprium/Hetonite Component's own
Nonzero Production tiers key DIRECTLY on their real
`xiranite_component_item`/`cuprium_component_item`/
`hetonite_component_item` resources (now real in 1.4, unlike 1.2e)
instead of needing a redundant per-component `*_flow` dimension the way
1.2e's `pp_goals.py` did -- only the AGGREGATE "any component nonzero"
goal still needs `component_flow`, since no single real resource means
"any of these five". Pyrrolite Component (`gearing_unit`) needed no
such fix either -- it was already a real resource from the start.
**Real bug found and fixed while building this**:
`xiranite_component_item`/`cuprium_component_item`/
`hetonite_component_item` had no `RESOURCE_BELT_SPEED` entry in
`wuling_1p4.py` at all, silently making them invisible to delivery-quota
candidates and depot-accumulation tracking (both require
`belt_speed == 30.0` to recognize a "real solid item") -- fixed by
adding all three at 30.0, matching every other named Component/Part.
14 tests in `tests/endfield/test_pp_goals_1p4.py`, all passing.

**`src/factorylib/endfield/refine.py`**: added `refine_1p4()` alongside
the existing `refine()` (a new parallel function, not a generalization
of `refine()` itself, to keep zero risk to its own tests). Takes
`(base_result, base_formula_names)` instead of one `SearchResult`,
since `wuling_1p4.search()` returns a plain `(OptimizeResult,
list[str])` tuple (no z/metatransfer scalar bookkeeping -- every
discrete choice is an ordinary named formula rate in 1.4). Uses the
now-generalized `goals._plan_complexity` with 1.4's own
`RESOURCE_NAMES`/`RESOURCE_BELT_SPEED` and a bookkeeping predicate
combining `wuling_1p4.is_forge_or_metatransfer_formula` +
`pp_goals.is_pp_bookkeeping_formula`. 5 tests in
`tests/endfield/test_refine_1p4.py`, all passing.

**`src/factorylib/endfield/wuling_1p4.py`**: also added `GOOD_YIELD`
(extends 1.2e's with `pyrrolite_part_sell`/`separator_core_sell`/the
new continuous-recipe formulas), `POWER_YIELD`/`METATRANSFER_ITEMS`
(reused unchanged, no new power routes or metatransfer options in 1.4),
`SELL_PRIORITY` (1.2e's order + the two new sellable goods appended),
`SECONDARY_GOAL_FORMULA_NAMES`/`SECONDARY_PLUMBING_FORMULA_NAMES`
(reused unchanged -- incomplete for 1.4's own new zero-$ intermediates,
noted as a known gap, not a correctness bug). **Real bug found and
fixed**: `WulingConfig1p4._v1p2e_config()` truncated `base_supply` to
1.2e's shorter length before delegating, but NOT `metatransfers` --
passing a full 1.4-length `--metatransfer` vector (as the CLI's own
help text tells you to) crashed with a numpy shape-mismatch inside
`v1p2e.build_formulas()`. Fixed by truncating metatransfers the same
way. Caught by `test_main_explicit_purify_building_and_metatransfer_flags`.

**`src/factorylib/endfield/delivery.py`**: generalized
`accumulation_rates()` the same way `goals._plan_complexity` was
generalized last session -- `resource_names`/`resource_belt_speed`/
`resource_labels`/`good_yield`/`formula_labels`/`stashable_good_formulas`
are now optional keyword parameters, defaulting to 1.2e's own (so
existing callers/tests are unaffected). 1.4-specific note: unlike 1.2e
(where all four Gear Components are pure dead ends), only
`ferrium_component` has no real consumer in 1.4 -- Xiranite/Cuprium/
Hetonite Component now feed the Crafting Point chain, so their surplus
correctly comes through `resource_slack` instead (now that they have a
`RESOURCE_BELT_SPEED` entry, see above), matching how `sandleaf_powder`
was already fixed once *it* gained a real consumer.

**`src/factorylib/endfield/cli.py`**: now imports `wuling_1p4`/
`pp_goals_1p4`/`refine_1p4` by default (aliased to the same names the
file already used, e.g. `WulingConfig1p4 as WulingConfig`) instead of
1.2e's `wuling`/`pp_goals`/`refine`. Added `--continuous-thresholds`
(store_true) wired to `WulingConfig1p4.continuous_thresholds`. Removed
`_format_forge_allocation` (1.2e's `z` scalar doesn't exist in 1.4 --
Forge of the Sky's 3-way allocation already shows via the ordinary
per-formula listing, using its own `FORMULA_LABELS` entry) but kept
`_format_metatransfer`, now driven by scanning each `metatransfer_option_i`
formula's own rate (since there's no single scalar to key off either) --
a bare `metatransfer_option_0` name in the formula listing wasn't
informative enough on its own to drop this. Added `gearing_unit`
(Pyrrolite Component) to the Gear-Component summary block, with a
`_GEAR_DISPLAY_NAMES` override so it doesn't show its full verbose
recipe description in that terse context.

**`tests/endfield/test_cli.py`**: 13 pre-existing tests were hardcoded
to 1.2e-only assumptions (its own historical $ figures, the `z=`/
metatransfer print format, 40-length `--base-supply`/`--metatransfer`
vectors, `fake_refine` mocks matching the old `refine()` signature).
All updated to 1.4's own real, empirically-verified numbers/behavior --
notably, 1.4's default scenario has its OWN genuine Forge-of-the-Sky
allocation tie (a 3-plain/9-Stable-ENV split and an all-Stable-ENV 12/0
split are economically identical), which replaced the old ya/jincao_tea
substitution tie test (that substitution still exists in 1.4, but
neither `ya` nor `jincao_tea` specifically happen to be part of the
$-optimal baseline any more -- `yc`/`jincao_drink` cover that tiny
sliver instead, so banning all four Yazhen/Jincao-family sell formulas
together, not just two, is needed to see any $ change). One test
(`test_main_no_ties_when_jincao_substitute_is_banned`) was removed
outright -- its premise (an isolated no-tie scenario once the zero-$
secondary-goal degeneracy is excluded) doesn't hold in 1.4's much larger
economy, which has other genuine ties (the forge split above) that
persist regardless of which sell formulas are banned; the underlying
exclusion code (`_tie_detection_exclude`, `SECONDARY_GOAL_FORMULA_NAMES`/
`SECONDARY_PLUMBING_FORMULA_NAMES`) is untouched by this session's
changes, so the regression risk it guarded against is unchanged.

**Verification**: full suite (3136 tests) passing, `ruff check`/`ruff
format --check` clean, `python -m factorylib.endfield` (both with and
without `--continuous-thresholds`) run end-to-end and manually inspected
-- Optimal/Most-fit/Power/Delivery-quota/Gear-Component/Delivery-job-
prediction/Diagram sections all print sensibly for 1.4.

**Not done in this pass** (flagged, not forgotten):
- `--continuous-thresholds` only affects `WulingConfig1p4` -- no
  dedicated CLI test exercises the flag itself yet (manually verified
  it runs and changes the reported $ figure as expected).
- `SECONDARY_GOAL_FORMULA_NAMES`/`SECONDARY_PLUMBING_FORMULA_NAMES`
  still don't cover any of 1.4's own new zero-$ intermediates (Cuprium
  Canister, Separator Core, the Crafting Point chain, etc.) -- only
  affects which "tied alternatives" get surfaced/suppressed, not
  correctness.
- No Graphviz diagram layout changes for 1.4's much larger formula set
  -- `generate_diagram` is fully generic (parameterized by
  resource/formula names+labels), so it works, but hasn't been visually
  inspected for legibility at 1.4's scale (this environment has no
  `dot` binary installed to render an actual image with).

## Virtual power/Water/Acid $ tax (WulingConfig1p4.power_dollar_tax)

Confirmed with the user: the raw $-maximizing `search()` has no concept
of power at all, so it was genuinely indifferent between wasteful and
efficient recipe choices that only differ in power/Carbon/Water/Acid
usage (e.g. Forge of the Sky's Carbon-sourcing split -- see "Two-layer
integer model" section above). Fixed via a real (if deliberately
partial) $ tax: every formula in `wuling_1p4.FORMULA_WATTS` (Planting
Units' direct power draw, plus every OTHER formula's implicit Water/Acid
draw per old_prompt.md's Fluid Pump ratios: 5W/60Water, 10W/60Acid) gets
a small negative `Formula.output`, converted from Watts via SC Wuling
Battery's own real $/W exchange rate (`$54/item ÷ (3200W/1.5 items) =
81/3200 $/W`, not an arbitrary epsilon). `search()` backs the tax back
out of the reported `dollar_output` (`power_dollar_tax_paid()`) before
returning, and `cli.py`'s tied-alternatives section does the same --
confirmed with the user the tax's own $ amount has no real-world meaning
(it's a deliberately incomplete power model), only its RELATIVE effect
on the solver's vertex choice matters. Defaults to True;
`--no-power-dollar-tax`/`power_dollar_tax=False` restores the old
indifferent behavior (used by historical-reproduction tests and by
`test_main_prints_discrete_branch_ties`/`test_main_prints_alternatives_
section_when_tied`, which specifically re-exercise the tie the tax now
resolves).

**Real engineering wrinkle, resolved**: `Formula.output` can't be
negative at CONSTRUCTION time (`factorylib.optimize.Formula.__post_init__`
raises), but direct attribute MUTATION after construction bypasses that
check entirely (confirmed: `f.output = -5.0` after `Formula(...)`
succeeds) -- exactly the same "last write wins" pattern
`config.formula_limits`/`formula_outputs` already use. This avoided
touching `factorylib.optimize.py`'s own tested invariant at all. An
earlier draft tried routing everything through a synthetic
`raw_dollar`/`dollar_conversion` resource+formula instead -- fully
reverted once this simpler mutation approach was found; it was causing
real bugs (leaked into `refine_1p4`'s initial search state, oversubscribing
a zero-supply resource; polluted the CLI's income breakdown with a huge
fake "sold good"; triggered a genuine HiGHS numerical-scaling "unbounded"
misdetection when combined with `formula_outputs={"pyrrolite_part_sell":
1e9}` in a reachability test).

Also drives `pp_goals_1p4.py`'s own (separate, always-on, NOT gated by
`power_dollar_tax`) `power_flow` consumption for the same `FORMULA_WATTS`
formulas -- confirmed with the user: "adding more planting units
increases the [effective 7000W] power goal" should be real and visible
in the refined "Most fit" section's own Power accounting, independent of
whether the raw-$-layer tie-break tax is enabled.

## `_threshold_formulas` simplified: one formula-name set, not two

Earlier design: `continuous=True` returned a single bare-name formula,
`continuous=False` (default) returned `{name}_alloc`/`{name}_run` --
meaning the formula-NAME SET itself depended on
`WulingConfig1p4.continuous_thresholds`, forcing every caller that needed
one of these 22 formulas by name (the power tax, `pp_goals_1p4`'s
power_flow loop, `power_dollar_tax_paid`) to guess which variant existed
via `name if name in f else f"{name}_run"`. Simplified per the user's
suggestion: ALWAYS build the `{name}_alloc`/`{name}_run` pair; `continuous`
just toggles `{name}_alloc`'s `integer` flag (True=default, real building
count; False=continuous "fractional buildings", mathematically exact
reproduction of the old proportional-folding approximation, verified:
same $2183.9459 continuous-mode figure as before the refactor). Removes
all three `actual_name = ...` workarounds entirely. `_THRESHOLD_RECIPES`
also now auto-derives all 11 reverse recipes from the 11 forward ones
(`_reverse_threshold_recipe`: sign-flip every entry in
`other_consumption`, verified against all 11 real reverse recipes
including both self-referential ones) instead of hand-duplicating 22
tuples -- removes the transcription-mismatch risk entirely, per the
user's own suggestion.

## Real bug found and fixed: Delivery quota vs. Delivery job prediction contradiction

User-reported: "Delivery quota: 210% of 2 jobs/day goal" (with several
materials listed as contributing) directly contradicted "Delivery job
prediction ... (nothing accumulates unconsumed in the depot)" immediately
below it. Root cause, confirmed empirically: `pp_goals_1p4`'s
`delivery_quota_from_X`/every Nonzero-Production-Goal `pp_*` tier formula
(`is_pp_bookkeeping_formula`-matched) scores a plan by literally
CONSUMING the real resource in the LP (e.g. `delivery_quota_from_
yazhen_powder` draws `box_capacity/1440` yazhen_powder/min directly) --
correct for scoring purposes, but it means that consumption is already
baked into `resource_slack`, so a material fully "spent" satisfying its
own pp tier shows zero slack even though nothing physical actually
consumed it (confirmed: `yazhen_powder`/`jincao_powder`/`carbon`/
`carbon_powder` all showed exactly 0.0 slack despite each contributing
"1 quota"). `cli.py`'s "Delivery job prediction" section was feeding
`accumulation_rates()` this SAME (already-reduced) slack, hiding those
materials from the depot simulator entirely. Fixed: `cli.py` now computes
a separate `delivery_display_slack` excluding every
`is_pp_bookkeeping_formula`-matched formula's own consumption, used only
for `accumulation_rates()` -- confirmed empirically this recovers the
true surplus (yazhen_powder/jincao_powder/carbon/carbon_powder all show
exactly 8.33/min = 1 box's worth once excluded; hetonite_part shows
0.5/min matching its own pp tier). The "Warning: this solution fully
saturates X" headroom check and the material-balance display correctly
keep using the ORIGINAL (unfiltered) slack, since those ARE about the
LP's real resource balance from the solver's own perspective.

## FORMULA_LABELS rewritten: real building name + explicit /min quantities

User-reported: labels like "Yazhen Planting (→ Yazhen)", "Cuprium Ore
Refining (Cuprium Ore → Cuprium + Sewage)", "Ferrium Bottle Moulding
(Ferrium → Ferrium Bottle)" hid both which real building runs a recipe
and how much of anything it actually needs. Rewrote essentially every
production-formula label in `wuling_1p4.FORMULA_LABELS` (both the
1.2e-inherited ones, overridden here -- `wuling.py`'s own labels stay
untouched -- and every 1.4-specific one) to the format `"{Building}:
{qty}/min {item} [+ ...] → {qty}/min {item} [+ ...]"`, e.g. "Moulding
Unit: 60/min Ferrium → 30/min Ferrium Bottle". Every quantity verified
against each formula's own `make_formula()` call in source (not retyped
from memory), including `FORMULA_WATTS`' implicit Water/Acid quantities
and `_THRESHOLD_RECIPES`' `[threshold 6/min]` tags. Left `_sell`
formulas ("X (sellable)") and the Crafting Point chain (no real
in-game building backs those conversions) in their simpler existing
style. `v1p2e.FORMULA_LABELS` is still spread in first as a fallback
safety net for anything not explicitly overridden.

Also extended `_GEAR_DISPLAY_NAMES` (cli.py) to cover all 5 Gear
Component formula names (previously only `gearing_unit`) -- the terse
per-line Gear Component summary loop started printing full verbose
recipe descriptions once FORMULA_LABELS grew long, this override table
keeps that specific display terse.

## BOM/recipe-listing deduplication + Material Balance multiples annotation

User-reported: with the Income breakdown section now showing full
per-sell-formula detail (amount, % produced, % goal, sold vs.
accumulating), the plain per-formula recipe listing directly above it
was repeating the exact same sell formulas with strictly less detail
(just a bare rate). Fixed: `_format_result()` now computes
`dollar_contribution_names` (via the existing `_dollar_contributions()`
helper, shared with the income breakdown itself) and skips those names
in the main listing loop -- but ONLY when the income breakdown is
actually shown (`formulas is not None and stock_bill_cap is not None`);
"Tied alternatives" calls (`formulas=None`) are unaffected, since there's
no breakdown there to be redundant against and they still need to show
which goods differ between tied alternatives.

Separately, user asked why a line like "+120/min from Refining Unit:
30/min Ferrium Ore → 30/min Ferrium" wasn't confusing against the
label's own "30/min" figure -- it was, since the label states the
per-multiple ratio but the balance line shows the aggregate flow with no
multiples count to bridge them. Fixed: `_format_material_balance()`'s
source/sink tuples now carry the formula's own `rate` as a third element
(`None` for the base-supply line, which has no formula/rate behind it),
appended as `" (N = N.NNNN multiples)"` on every formula-backed line.

Regression tests added in `tests/endfield/test_cli.py`:
`test_main_material_balance_lines_show_formula_multiples`,
`test_main_recipe_listing_does_not_duplicate_income_breakdown_entries`.

## Flagged, NOT yet fixed: sc_sell "net: 0/min" vs. allocate_by_priority "accumulating $" inconsistency

User noticed the SC Wuling Battery example prints BOTH "net: 0/min" in
Material Balance (the LP's `sc_sell` formula fully consumes every
battery produced -- nothing physically left over, by the LP's own
resource-balance accounting) AND "accumulating: 796.5 $/min" in the
Income breakdown (implying real unsold, physically-piling-up batteries).
Root cause, confirmed by reading `factorylib/priority_sell.py` in full:
`allocate_by_priority()` is a PURE post-hoc, display-only greedy
allocator over each sell formula's already-computed $ contribution --
it has no connection to the LP's own resource balance at all, and its
own module docstring's framing ("whatever doesn't fit in the budget
simply accumulates unsold") directly implies real leftover inventory
that Material Balance's "net: 0" denies exists for that same good. This
"accumulating" $ figure also gets converted back into an item rate and
fed into the delivery-job simulator (`cli.py`'s `refined_unsold` loop),
so it's not just a cosmetic double-narrative -- it currently makes
"unsold" SC Wuling Batteries eligible to be selected as delivery-job
cargo, which arguably shouldn't be possible if the batteries were
genuinely fully consumed making room for `sc_sell`'s own rate.

**Deliberately not fixed** -- this was surfaced as a discovery/
explanation in response to a direct question, not a request to change
behavior, and per "at least flag suspicious changes" this is being
recorded rather than silently patched. Needs a decision on the intended
semantics before touching it: either (a) `sc_sell`-style formulas should
never be treated as "fully consuming" their input if some of their own
$ output can't actually be sold yet (i.e. genuinely throttle the LP's
own `sc_sell` rate down to what `stock_bill_cap` can absorb, making
Material Balance and the income breakdown agree that leftover batteries
exist), or (b) the "accumulating" framing itself is wrong and should be
reworded to something like "produced-but-not-yet-billed $" without
implying physical inventory, and should NOT feed the delivery-job
simulator as a real material candidate.

## Regression tests added (2026-07-18/19 session, per "start adding tests now")

Per explicit instruction ("if you're done with the 1.4 scenario, you
should start adding tests now to catch regressions, or at least flag
suspicious changes"), added dedicated coverage for every mechanism
introduced/changed this session that had none before:

`tests/endfield/test_wuling_1p4.py` (+7 tests):
- `test_power_dollar_tax_resolves_forge_split_tie_deterministically` /
  `test_power_dollar_tax_disabled_restores_old_tie` /
  `test_power_dollar_tax_never_changes_the_true_optimal_dollar_value` --
  the tax's whole reason to exist (breaking the Forge-of-the-Sky tie)
  and its core invariant (never altering the reported `dollar_output`)
  are both now regression-guarded.
- `test_power_dollar_tax_paid_matches_manual_computation` /
  `test_power_dollar_tax_paid_zero_for_no_rates` -- direct arithmetic
  check on `power_dollar_tax_paid()`, independent of any LP solve.
- `test_formula_watts_formulas_get_negative_output_when_tax_enabled` --
  confirms the actual `Formula.output` mutation mechanism (the
  post-construction-mutation loophole) fires for every `FORMULA_WATTS`
  entry, resolving to `{name}_run` for threshold recipes.
- `test_continuous_thresholds_does_not_change_formula_names` /
  `test_continuous_thresholds_toggles_alloc_integer_flag_only` --
  guards the core invariant of the `_threshold_formulas` simplification
  (formula-name set never depends on `continuous_thresholds`, only
  `.integer` does).
- `test_threshold_recipes_reverse_derivation_matches_hand_verified_values`
  -- spot-checks `_reverse_threshold_recipe`'s sign-flip against the
  hand-verified values for a plain recipe, a batch recipe, and both
  self-referential recipes (fluid_gas_xiragen/solid_gas_xiragen), so a
  future sign-flip bug in the auto-derivation can't slip in silently.

`tests/endfield/test_cli.py` (+3 tests):
- `test_main_delivery_quota_contributors_are_not_hidden_from_delivery_job_prediction`
  -- reproduces the exact scenario the user originally reported
  (materials fully committed to their own delivery-quota bookkeeping
  formula used to vanish from "Delivery job prediction" entirely) and
  confirms every material listed under "Delivery quota" now also
  appears as a real delivery-job candidate.
- `test_main_material_balance_lines_show_formula_multiples` /
  `test_main_recipe_listing_does_not_duplicate_income_breakdown_entries`
  -- cover the BOM dedup and Material Balance multiples annotation
  described above.

Full suite (3146+ tests) passing, `ruff check`/`ruff format --check`
clean after these additions.

## Real bug found and fixed: "1 multiple" wasn't the recipe's real /min rate

User-reported: "Forge of the Sky, Stable ENV: 1/min Carbon + 1/min Water
→ 1/min Xiranite" was wrong -- the real recipe is "every 2 seconds" =
30/min at full building utilization. Root cause: every threshold
recipe's `{name}_run`, `xi_forge_run`/`xi_forge_stable_env_run`,
`gas_reactor_globe_run`, the two Stable-ENV Purification Unit variants,
and every plain new-1.4 recipe (fitting_unit/moulding_unit/
gearing_unit/filling_unit_*/packaging_unit/
reactor_crucible_liquid_heavy_xiranite/purification_heavy_xiragen/
purification_hetonite_gas) left "1 multiple" at the raw per-cycle item
count instead of the real per-building rate -- inconsistent with every
other production formula's "1 multiple = 1 building" convention (e.g.
1.2e's own hx_make: "60 Xiranite -> 6 Heavy Xiranite"). Fixed by
minting exactly 1 unit of capacity per alloc-building (was max_rate
units) and scaling each run/plain formula's consumption by max_rate (30
for "every 2 seconds", 6 for "every 10 seconds") to match -- verified
by hand this is a pure change of variables (same threshold_good-per-
output ratio either way), confirmed empirically the full suite stayed
green with zero assertion changes (no $-optimal figure moved).
FORMULA_WATTS, GOOD_YIELD, pp_goals_1p4's `_COMPONENT_FLOW_CONTRIBUTORS`,
and every affected FORMULA_LABELS entry updated to match. Regression
tests added in `test_wuling_1p4.py`.

Also renamed every `_THRESHOLD_RECIPES` capacity's RESOURCE_LABELS entry
from the generic "(Building) (Item[, reverse]) Capacity" template (e.g.
"Fluid-Gas Transmuting Unit (Aquagen, reverse) Capacity", which didn't
say what it actually converts) to "X → Y Threshold Activations" (e.g.
"Aquagen → Water Threshold Activations"), naming the real conversion
direction directly.

## CLI output rewritten as Markdown

User request: the tab/indent report "looks bad when pasted elsewhere."
Rewrote `cli.py`'s entire report as Markdown -- `##` headers for major
sections (Optimal solution, Material balance, Tied alternatives, Most
fit solution found, Delivery job prediction), `###` for each
Alternative/Refined solution sub-block, nested `"- "` bullets
(`_bullet(depth, text)` helper) for everything else instead of raw
tab-stop indentation. `_format_result`/`_format_income_breakdown`/
`_format_material_balance` all take an explicit `header_level`/`depth`
now. All ~46 test_cli.py tests updated to the new format (mostly
substring assertions that didn't need to change; two needed the
"### Refined solution" \\n "dollar = ..." two-line structure instead
of the old single-line "Refined solution: dollar=...").

## Inergen/Xiragen base supply bumped (data sheet update)

kaneko_1p4_data_sheet.md's "New Max Raw Material Income" revised
Inergen from "at least 260/min" to "at least 460/min" and Xiragen from
30/min to 100/min (Cuprium Ore's 420 was already current). Updated
`DEFAULT_INERGEN`/`DEFAULT_XIRAGEN` in wuling_1p4.py to match. This
shifted the default scenario's own $-optimal figure ($2171.10 ->
$2426.9080, `606727/250`) and, separately, made the Forge-of-the-Sky
Carbon-sourcing tie no longer reproducible from the DEFAULT scenario
even with `power_dollar_tax=False` (the much larger Inergen/Xiragen
supply gives the all-Stable-ENV route an independent, non-tax reason to
win outright) -- `test_power_dollar_tax_disabled_restores_old_tie` now
constructs a reduced-supply config explicitly (Inergen=260, Xiragen=30)
to keep exercising that specific tie in isolation.

## tmp_notes cleanup

Deleted 4 fully-superseded scratch files (contents already absorbed
into this file's own narrative, and none cited by source code):
`future_work.md` (SA-replacement modularity requirement, see "Carried-
over / deferred scope" above), `clarification_delivery.md` (delivery
box capacity 14k->12k, already applied), `adjust_power_curve.md`
(power hard-cap retune, already applied), `sample_out.md` (a stale CLI
output capture, obsolete after the $ figure and format changed twice
since it was written). Kept everything still cited by source code
(`1p4_new_features.md`, `new_goals.md`, `old_prompt.md`,
`flexible_gear_crafting.md`, `power_consumption.md`,
`make_plants_not_free.md`) plus live/unresolved notes
(`investigation_questions.md`, `clarification_dollar_goal.md` -- the
latter explicitly says "this is still not the final number") and the
durable `ai_checklist_before_merge.md` checklist.
