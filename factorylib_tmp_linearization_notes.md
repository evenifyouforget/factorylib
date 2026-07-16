# Linearization experiment — progress notes / todo

Goal: replace the nonlinear Part-4 fitness function (goals.py's
_threshold_term shaped rewards) with a pure-LP "prosperity points" (pp)
system per factorylib_tmp_linearization.md, so the *entire* allocation
decision (which formulas to run, at what rate) can be solved exactly by
maximize_dollar-style LP, with SA only needed afterward (if at all) for
fraction-denominator simplicity.

## Design understanding (derived, not yet implemented)

Each "goal" in the plan becomes a set of parallel small Formulas, each
consuming a slice of some *virtual flow resource* (dollar_flow,
power_flow, delivery_flow, hetonite_part_flow, component_flow +
per-Component flows) and producing "pp" (repurposing Formula.output,
since pp fully replaces $ as the LP objective in this experiment — the
original $-earning formulas' output must be zeroed and replaced with
dollar_flow production instead).

Key mechanism: for a concave (diminishing-returns) piecewise-linear
utility of one variable, you don't need integer "which segment" logic —
stacking N parallel LP activities, each with slope pp/input strictly
decreasing across the stack and its own cap (limit), causes a
maximizing LP to *automatically* fill the steepest-slope activity first,
then the next, etc. Verified this against the plan's own numbers: every
listed tier group has strictly decreasing slope in the order given
(confirms the plan's tiers are self-consistent, no reordering needed).

Tiers transcribed as (input_per_multiple, pp_per_multiple, max_multiples):
- Sellable goods (dollar_flow, $/min): (1090,1000,1) (1090,100,0.2) (1090,50,0.8) (1090,10,inf)
- Power (power_flow, W): (7000,1000,1) (7000,100,0.05) (7000,10,inf)
- Delivery (delivery_flow, items/day — sandleaf_powder only for v1): (28000,1000,1) (28000,100,0.5) (28000,10,inf)
- Hetonite Part (hetonite_part_flow, items/min): (0.1,500,1) (1,100,1)
- Generic Component (component_flow, items/min, sum of 4 Components): (0.1,500,1) (1,100,1) (1,10,inf)
- Each of Xiranite/Cuprium/Hetonite Component (own flow, items/min): (0.1,100,1) (1,20,1)
  — note: a unit of e.g. Xiranite Component counts toward BOTH generic
    component_flow AND xiranite_component_flow simultaneously (byproduct
    of the same real formula's rate) — this is what creates the implicit
    "diversify across Components" incentive without explicit priority
    weights.

## Plan

1. [ ] Toy sanity check: 1-resource, N-tier LP confirms slope-ordering
   claim (steepest tier fills first, no artificial reordering needed).
2. [ ] Build `build_linearized_formulas()`: extends the real 48-formula
   Wuling set with virtual flow byproducts (dollar_flow/power_flow/
   delivery_flow/hetonite_part_flow/component_flow*4), zeroes $ output
   on sc/lc/hp_sell/hx_sell/ya/yc/jincao_tea/jincao_drink/xi_sell/cp_sell.
3. [ ] Add the ~15 pp-tier formulas per the table above.
4. [ ] Solve via plain maximize_dollar (pp as objective) — no SA needed
   for this part at all.
5. [ ] Compare resulting plan (real $ recovered, power, delivery good,
   Component diversity) against the existing nonlinear-fitness + SA
   refined solution, qualitatively and numerically.
6. [ ] If promising: check whether *fractions* in the pure-LP solution
   are already reasonably simple on their own (no complexity term at
   all yet) before deciding whether SA is still needed as a second pass.
7. [ ] Write up findings, recommend whether to pursue as a real feature.

## Status log

- [x] Step 1 (toy sanity check): confirmed. 4-tier sellable-goods ladder
  with supply=5000 fills tier1(1090 cap)->tier2(218 cap)->tier3(872
  cap)->tier4(unlimited) in exactly that order, total pp=1085.87,
  matching hand computation. Also confirmed partial-tier and zero-supply
  behavior is correctly linear, no reordering artifacts. LP mechanism
  for concave piecewise-linear utility via parallel capped activities
  works exactly as expected, no integer/binary variables needed.
- [x] Step 2-4: prototype at
  /tmp/claude-1000/.../scratchpad/linearize_prototype.py (not committed
  anywhere real -- lives in the session scratchpad). Extends the real 48
  -formula Wuling set with 8 virtual flow resources, zeroes $ output on
  the 10 $-earning formulas (replaced by dollar_flow production), adds
  power_flow/delivery_flow/hetonite_part_flow/component_flow*4
  byproducts, adds ~17 pp-tier formulas from the plan's table. Solved
  with plain maximize_dollar -- confirmed a genuine bug caught early:
  delivery target (28000/*day*) wasn't unit-converted against
  sandleaf_powder's own */min* rate, silently under-filling the delivery
  ladder by a factor of 1440. Fixed (divide by 1440).
- [x] Step 5 (compare against current nonlinear+SA default): stark,
  clear difference --
    - pp-LP: $806/min real (57% of the 1416 $-only optimum), power
      6147 W (88% of 7000 target), Xiranite/Cuprium/Hetonite Components
      all running (diversified, matching the plan's "own-flow" tiers'
      intent), Ferrium Component still 0 (no own-flow tier for it in the
      plan -- only feeds the generic component_flow ladder, so it never
      gets the same steep initial incentive the other three do).
    - current default (nonlinear fitness + SA, complexity_weight=1.0):
      $1092/min, power=0 W, ALL FOUR Components=0.
  The pp-LP approach makes an explicit, principled trade (sacrifice 28%
  of raw $ to substantively hit power + gear diversity goals) that the
  current default nonlinear weights don't produce at all -- this isn't
  a mechanism failure on the old side, it's that gear_importance/
  power_importance are tuned too low by default to overcome the $
  incentive (a finding from earlier in this session too). The pp tiers
  make this trade-off fully transparent/tunable via a plain number
  table instead of an opaque weighted-nonlinear-term balance.
- [x] Step 6 (fraction simplicity check, no SA at all yet): as the plan
  itself warned up front, denominators are messy (up to 793; total
  simplicity.fraction_complexity = 300.5 across the real-recipe
  formulas) -- much worse than the current SA-refined solution's
  typical ~120-150. Confirms the linearization genuinely captures
  "which formulas, how much" via exact LP, but does NOT touch fraction-
  denominator simplicity at all, exactly as scoped. A second pass
  (existing SA round_down/round_up/allocate_slack moves, or a smaller/
  faster SA restricted to *only* denominator cleanup since the pp-LP
  already picked the allocation) would still be required for a
  physically-buildable recommendation.

## Conclusion / recommendation

The core idea works exactly as hoped: a stack of parallel, capped,
strictly-decreasing-slope LP "conversion formulas" faithfully reproduces
a concave piecewise-linear (diminishing-returns) reward with zero
nonlinear solver machinery -- the entire "which formulas to run, at what
rate" decision becomes a single deterministic LP solve, replacing
goals.py's shaped nonlinear terms + the SA search's job of finding that
allocation. This is a real, valuable simplification if pursued: faster
(no SA needed for allocation), deterministic/reproducible, and the
goal trade-offs become plain numbers in a table instead of buried in
threshold-shape math.

What it does NOT replace: fraction-denominator simplicity is still a
separate, necessarily-nonlinear concern (as the plan says up front) --
SA (or some other combinatorial search) would still be needed as a
second stage, now with a *much smaller job* (just simplify fractions
around an already-optimal allocation, not search the allocation too).

Not yet explored (would need another pass, flagging rather than doing
speculatively):
  - Tuning the tier VALUES themselves (the plan's own numbers are a
    first draft the user invited me to adjust) -- e.g. giving
    ferrium_component its own-flow tier for parity with the other 3
    Components, or reweighting the sellable-goods ladder so the $
    sacrifice is smaller.
  - Whether running the existing SA moves *on top of* the pp-LP's fixed
    allocation (fraction cleanup only, not reallocation) recovers good
    fractions without giving back the $/power/component gains.
  - "any material -> Generic Material" for delivery (v2 wires
    sc_battery/lc_battery/sandleaf specifically -- see below -- but a
    fully general "any leftover physical good" version isn't built).
  - Turning this into a real, tested module/CLI backend if the user
    wants to pursue it -- this file's prototype is scratch-only.

## v2 correction (real bug the user caught)

v1 above let a single formula grant MULTIPLE pp-tier byproducts "for
free" from the same rate -- e.g. hp_sell earned both dollar_flow AND
hetonite_part_flow simultaneously, as if selling a Hetonite Part for $
is *also* free crafting-ingredient credit; sandleaf_powder's *gross*
rate fed delivery_flow directly instead of its *net* surplus after
ori_to_dop/packed_origocrust_make/dense_ferrium_powder_make take their
share. Real physical goods have exactly ONE destination (sold, used for
power, held for crafting, or packed for delivery), never several at
once -- the same battery can't be both sold and consumed for power.

Fix: competing uses must draw from the SAME real resource pool, not a
virtual "byproduct." Since the real model already tracks sc_battery/
lc_battery/hetonite_part/sandleaf as genuine resource dimensions (the
battery-to-power split added to the real src/ tree this session made
this the natural approach), the pp tiers for delivery/crafting-ingredient
credit were rewired to consume those REAL resources *directly*, so they
structurally compete with sc_sell/hp_sell/ori_to_dop/etc. for the exact
same fixed supply -- no separate "hold for X" intermediate formula even
needed; verified sc_battery's production (14.7/min) exactly equals
sell+power+delivery consumption (0+3.28+11.42), zero double-counting.

Corrected numbers (same default scenario): $535/min real (was $806,
since delivery no longer double-dips battery production for free),
power hits the FULL 7000 W target exactly (was 6147, 88%) since power
now genuinely competes with -- rather than coexists for free with --
delivery and selling. Components: Xiranite 0.1/min, Cuprium 1.1/min,
Hetonite 0.1/min, Ferrium still 0 (same reason as v1: no own-flow tier
for it in the plan). The qualitative conclusion is unchanged (the LP
mechanism works, fractions still need a second SA pass), but the
specific $-vs-goals trade-off is now measured honestly rather than
inflated by free double-counted byproducts.

This file + the scratch prototype can be deleted before the final PR
per the user's own instructions; nothing here is wired into the real
src/ tree.

## v3: reusable tier generators + integer 100%-bonus (done)

Replaced v2's hand-massaged per-tier tuple literals with 3 reusable,
math-driven generators (no per-tier magic numbers), per the user's
request:

- `satisfaction_tiers(target, soft_cap_ratio, hard_cap_ratio, ...)`:
  unifies what were originally going to be two separate "Hard
  Satisfaction Goal" / "Soft Satisfaction Goal" generators into ONE,
  parameterized by two ratio breakpoints (both expressed relative to
  `target`) -- the user's own simplification once they saw the two
  drafts converging on the same curve shape. `n_ramp_tiers` tiers
  front-load-partition `[0, target]` via geometric split/decay
  (`_geometric_split`/`_geometric_decay`, both reusable helpers); one
  more tier covers `(target, target*soft_cap_ratio]` continuing the
  same decay; a final tier covers
  `(target*soft_cap_ratio, target*hard_cap_ratio]` at a much steeper
  `tail_decay` -- and nothing exists past `target*hard_cap_ratio` at
  all (that tier's cap is finite, not infinite). Power ("hard
  satisfaction" -- DIGE's battery-balancer convention of ~5% max
  overshoot) uses soft/hard = 1.05/1.40; sellable goods ("soft
  satisfaction" -- selling down to 0 faster, tolerating outages) uses
  1.20/3.00; delivery filler uses 1.10/1.80 -- all just parameter
  choices on the one curve, not different shapes.
- `nonzero_production_tiers(...)`: for goals with no real target at all
  (Hetonite Part, the 4 Gear Components) -- front-loaded, absolute
  `first_cap`-anchored (not target-relative), unbounded tail.
- `hard_satisfaction_bonus(target, bonus_pp)`: answers the user's
  question "is a 100%-reached bonus implemented via integer formulas?"
  -- it wasn't, now it is. A single `Formula(integer=True, limit=1)`
  that consumes exactly `target` units of flow for a lump `bonus_pp`,
  with NO fractional credit (MILP integrality forbids 0.5 multiples) --
  a genuine discontinuity the smooth tiers can't express alone. Added
  for dollar/power/delivery (not the no-target nonzero-production
  goods, which have nothing to be "100%" of).
- All constants now come from a `PPGoals` dataclass (dollar/power/
  delivery targets + soft/hard cap ratios + tier counts), not bare
  literals in `build_linearized_formulas` -- verified via
  `verify_parameterization()`, which re-solves under several different
  `PPGoals`/`WulingConfig` combinations:
  - Varying `power_target` (3500/7000/14000) and `dollar_target`
    (545/1090/2180): real output tracks the target correctly.
  - Varying `power_hard_cap_ratio` (1.05/1.40/3.00): **no effect on the
    result** -- a real, useful finding, not a bug. With the default
    `tail_decay=0.05`, the tail tier's pp/unit is so small that the LP
    never finds it worth the real resource cost to reach, regardless of
    where `hard_cap_ratio` puts the ceiling -- the *soft* cap is the
    actual binding constraint in practice. `hard_cap_ratio` only starts
    to matter with a less aggressive `tail_decay`; not tuned further
    since this is exactly the kind of tradeoff the user said was fine
    to leave for later experimentation ("try tweaking these constants
    further").

## Related real-src/ fixes made in the same session (separate from this
## scratch experiment, but discovered through the same conversation)

While tuning the *real* (non-pp) nonlinear fitness function to match
the same priority ordering (power > sellable goods > delivery > gear)
and Hard/Soft Satisfaction philosophy discussed here:
- `WulingGoals` defaults reordered/retuned: `power_importance=50`,
  `stock_bill_importance=35`, `delivery_importance=15`,
  `gear_importance=8`; added `power_excess_importance=0.5` to decouple
  power's overshoot reward from its shortfall penalty (a "hard
  satisfaction" goal shouldn't reward overshoot just because missing it
  is heavily penalized).
- `complexity_weight` default 1.0 -> 0.1, `--refine-iterations` default
  2000 -> 6000, so the SA search reliably reaches power/delivery
  targets and produces at least some Gear Components instead of
  over-prioritizing fraction simplicity.
- Real bug fixed in `factorylib/delivery.py`: the depot simulator
  picked whichever material had the most and subtracted `box_capacity`
  unconditionally, even when that amount was *less* than
  `box_capacity` -- silently driving it negative instead of recognizing
  the delivery job failed. Now returns `DeliverySimResult(tally,
  failed_jobs)`; the CLI warns when jobs fail.
- Did NOT wire delivery-job failure into `fitness()` directly -- it's
  called thousands of times per SA search, and the full stochastic
  multi-day depot simulation is too expensive to run per candidate. The
  existing rate-based delivery threshold (already calibrated to the
  box_capacity floor) is the practical proxy, now strengthened via the
  reweighting above.

All 3086 tests pass, `ruff check`/`ruff format --check` clean, every
historical dollar invariant still exact (unaffected -- these changes
only touch the nonlinear fitness/search/delivery layers, not the raw
$-optimal LP).

## v4: Delivery Job Quota mechanic (real success, no fallback needed)

Replaced the hand-picked "sc_battery/sandleaf, each its own
satisfaction_tiers" delivery model with a much more faithful one, per
the user's suggestion: every *solid* resource (belt_speed 30 -- liquids
can't sit in a depot at all) gets its own capped conversion,
`_materialize_delivery_quotas()`:

    box_capacity (14000/day, -> /min) of material X -> 1 delivery_quota_flow
    (limit = delivery_quota_max_multiple, default 1.0 -- "at most one
    box's worth of quota from any single material")

Quota itself is then a Soft Satisfaction Goal (`satisfaction_tiers`)
anchored at `delivery_jobs_per_day` (2.0) -- "fully covering every job
today" is the 100% target, with a `hard_satisfaction_bonus` on top.

Rationale (the user's, from real gameplay): after a delivery job picks
whatever material has the most and drains `box_capacity` from it, that
same material is very unlikely to still be #1 for the *next* job --
covering multiple jobs/day for real requires multiple DISTINCT
materials, not one material supplying an unlimited amount. Capping each
material's own quota formula at 1.0 forces exactly that.

**Result: the LP found the multi-material split strategy entirely on
its own, no hand-coding required.** `sandleaf_plant` runs at rate 5.0;
the solver splits its output so some gets shredded into Sandleaf Powder
(partly consumed by `ori_to_dop`, the rest banked as one quota) while
some is deliberately kept RAW (unshredded Sandleaf) specifically to
bank as a *second*, distinct quota -- exactly the real-gameplay
strategy the user described ("produce 30/min Sandleaf, shred half,
resulting in 15/min Sandleaf + 45/min Sandleaf Powder"). It also pulled
in Ferrium Part (full quota) and partial Ferrium Powder as two more
distinct sources, reaching ~3.2 total quota (target was 2.0 + bonus).

Because this worked immediately, the two fallback plans the user
proposed weren't needed:
- Plan A ("let any solid contribute unlimited quota") -- not tried;
  the capped mechanic alone was already cheaper than expected.
- Plan B (partially cost plant production in W, since Planting Unit
  output looked "free") -- not needed; the LP differentiated between
  materials just fine using existing costs (Sandleaf's real downstream
  competition with `ori_to_dop`, Ferrium's shared use across several
  recipes) without an artificial power tax.

Not yet extended: Buckflower/Citrome/Aketine (the spec's other three
Planting Unit alternatives to Sandleaf) aren't modeled in `wuling.py`
at all -- adding them to this experiment would require building their
production chains first, out of scope here.
