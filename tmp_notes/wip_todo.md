# WIP / TODO notes (factorylib.endfield)

Scratch tracking file for work-in-progress across sessions. Not meant to
ship in a release PR long-term (delete once superseded by real issue
tracking), but committed for now per explicit instruction so it survives
between sessions.

## Where things stand

- PR #13 ("Specialize for Endfield: Endfield-specific multi-goal
  solving, integer formulas, CLI") merged to `main` at `fe099a3`.
- Current branch: `wuling-1p4`, for Endfield 1.4 content.
- Repo state on this branch: 3099 tests passing, `ruff check`/`ruff
  format --check` clean. Default CLI run (`python -m factorylib.endfield`):
  - Optimal ($-only): $1415.99/min (129.9% of $1090 stock-bill goal)
  - Most-fit (refined): $1125.47/min (103.2%), Power 9800W (140.0%,
    hard cap), Delivery quota 184.0% of 2 jobs/day, Cuprium Component
    200%, Xiranite Component 20%, **Hetonite Component 0%** (see below)

## Carried-over / deferred scope (from PR #13's own MR checklist)

- **Hetonite Component stuck at 0** -- confirmed genuine 3-way resource
  tension (dollar > 100% vs power at its hard cap vs Hetonite Component
  all compete for the same upstream Cuprium/battery-adjacent resource
  chain). Boosting Hetonite's own pp reward revives it but only by
  giving back the dollar/power gains. Was already marginal (3.4%)
  before the pp_decay retune that fixed dollar's stuck-at-100% bug.
  Needs dedicated recipe/resource-flow investigation (is there a
  cheaper real path to Hetonite Part that doesn't compete with SC
  Wuling Battery's Cuprium/Ferrium inputs?), not another weight nudge.
  **Revisit once 1.4's real recipes are in** -- new Hetonite routes
  (see below) may resolve this on their own, making further tuning of
  the *old* recipe set moot.
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

### Alternate gear crafting (Component substitution hierarchy)
Higher-tier Components can substitute for lower-tier gear recipes
(Hetonite Component -> Xiranite-tier gears, but not reverse). Pyrrolite
Component gets a 99% Wuling-Stock-Bill-cost discount, effectively
making it $-free and only Component-supply-limited -- per the notes
author's own read, this may make producing Xiranite/Cuprium/Hetonite
Component "completely unnecessary for endgame users" once Pyrrolite
exists. **Implication for our model**: once real numbers land, this
probably means each gear-tier's Formula should have one entry per
eligible Component type (Pyrrolite substitutable in for all 4 tiers,
Hetonite for tiers 1-3, etc.), all competing for the same $-maximizing
LP naturally -- no new mechanism needed, just more Formula entries,
same as how `sc_sell`/`hp_sell`/etc. already all compete for the same
Wuling Stock Bill cap.

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
