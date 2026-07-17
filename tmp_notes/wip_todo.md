# WIP / TODO notes (factorylib.endfield)

Scratch tracking file for work-in-progress across sessions. Not meant to
ship in a release PR long-term (delete once superseded by real issue
tracking), but committed for now per explicit instruction so it survives
between sessions.

## Where things stand

- PR #13 ("Specialize for Endfield: Endfield-specific multi-goal
  solving, integer formulas, CLI") merged to `main` at `fe099a3`.
- Current branch: `wuling-1p4`, for Endfield 1.4 content.
- Repo state on this branch: 3100 tests passing, `ruff check`/`ruff
  format --check` clean. Default CLI run (`python -m factorylib.endfield`):
  - Optimal ($-only): $1415.99/min (129.9% of $1090 stock-bill goal)
  - Most-fit (refined): $1178.63/min (108.1%), Power 7700W (110.0%,
    hard cap), Delivery quota 184.0% of 2 jobs/day, Cuprium Component
    200%, Xiranite Component 20%, **Hetonite Component 0%** (see below)
  - (numbers above reflect the power-curve retune and box-capacity fix
    applied this session -- see "Applied this session" below)

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
`xiranite_forge_alloc`/`heavy_xiranite_forge_alloc`:
```
Forge of the Sky: 2 Stabilized Carbon + 1 Water -> 1 Xiranite (every 2s)
Forge of the Sky: 10 Xiranite + 5 Xircon Effluent -> 1 Heavy Xiranite (every 10s)
```
Heavy Xiranite is now downstream of Xiranite (a real consumption, not a
separate capacity allocation) rather than the old two-parallel-outputs
framing. **Confirmed with the user**: global cap is 12 Forge of the Sky
buildings total (same number as 1.2e's `max_forges=12` default -- good
continuity), each committed to EITHER the Xiranite recipe OR the Heavy
Xiranite recipe, not both -- so the *shared-integer-pool, competing
consumers* mechanic from 1.2e (`forge_budget`) is still the right
shape, just feeding real recipe formulas instead of an abstraction
(and possibly literally still `max_forges=12` as the default, unless
1.4 changes it). Concretely: keep a
`forge_budget`-like virtual resource, with two `integer=True` Formula
entries (`xi_forge_alloc`, `hx_forge_alloc`) each consuming 1 unit of
it and producing the *capacity* to run the corresponding real recipe
(mirroring `wuling.py`'s existing `xiranite_forge_alloc`/
`heavy_xiranite_forge_alloc` pattern almost exactly -- this is closer
to a rename/rewire than a new mechanic).

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

## Open questions for the user (not yet asked)
- Once 1.4 recipes are confirmed, do we want a clean `wuling_1p4.py`
  (or a `config`-driven variant switch) alongside the existing 1.2e
  scenario, or should 1.2e be retired/replaced outright? Given how much
  of `wuling.py` is scenario-specific (RESOURCE_NAMES, RESOURCE_LABELS,
  build_formulas), this is a real fork-vs-replace decision worth
  raising before writing a lot of new formula code.
