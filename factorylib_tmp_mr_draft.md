- [x] All tests pass in CI (3099 passed locally, `ruff check`/`ruff format --check` clean)
- [x] Documentation is accurate (README's Graphviz/uv section checked against actual `--help` output)
- [x] Ran CLI with default parameters (`python3 -m factorylib.endfield`)
- [x] Max $ does in fact produce a lot of $ (206735/146 = $1415.99/min, 129.9% of the $1090 stock-bill goal)
- [x] Most fit solution from search is acceptable (default seed 0, `--refine-iterations` default 6000):
- [x] > 100% sellable goods (103.2% -- see below, this needed a real fix)
- [x] > 100% power (140.0%, at its hard cap)
- [x] Fulfills delivery jobs (184.0% of the 2 jobs/day goal)
- [ ] > 0 Hetonite Component -- **deferred**: genuine 3-way tension confirmed (dollar>100% vs power@hard-cap vs Hetonite Component all compete for the same upstream Cuprium/battery-adjacent resources); it was already marginal (3.4%) before any of this session's changes. Boosting its own reward revives it but at the cost of dollar/power/quota again -- needs dedicated recipe/resource-flow tuning, not a simple weight nudge. Follow-up.
- [x] > 0 Cuprium Component (200% of its 0.5/min informational reference)
- [x] > 0 Xiranite Component (20%)

Default seed (0) now satisfies 5 of 6 above; no seed search needed (the
one still-failing item is a structural tension, not seed-luck -- see
above).

**Real bug found and fixed while verifying this**: dollar output was
landing at *exactly* 100% of target, never more, across every seed and
iteration count tried (0..60000) -- confirmed via CLI testing this
looked wrong ("massively overcommits batteries to power... doesn't
produce any excess sellable goods"). Root cause: `satisfaction_tiers`'
`pp_decay=0.15` default left dollar's post-100% tier worth only ~3.4
pp/$, just under the ~6.7 pp/$ needed to ever outbid power's tail tier
for a unit of shared battery capacity (1.5 batteries -> 3200 W vs 6
batteries -> $324) -- so the search never claimed extra $ output even
though the $-only baseline shows it's physically available. Fixed by
raising the default to `pp_decay=0.20` (see `satisfaction_tiers`'
docstring in `pp_goals.py` for the full derivation); regression tests
added in `test_pp_goals.py` and `test_refine.py`.

**Also fixed during this pass** (see commit history for full detail):
- `factorylib.search`'s SA backend never enforced `Formula.integer`,
  letting the "reached 100% in one go" bonus formulas run at fractional
  rates -- this was the original cause of the power-overcommit symptom
  above, before the curve issue was found underneath it. Now enforced
  via `_integer_rates_valid`/`_toggle_integer_move`, with a final
  assertion as a safety net against future regressions.
- the delivery-job material picker included virtual bookkeeping
  resources (Forge Budget, Metatransfer Allowance) as if they were real
  depot-storable goods.
- `diagram.py` now falls back to writing raw `.dot` source when the
  system `dot` executable isn't installed, instead of silently doing
  nothing.

- [ ] tmp files deleted (`factorylib_tmp_*.md` files, including this one, still present -- delete before merge)
- [x] Varying each parameter results in the solution changing appropriately
- [x] Originium Ore income
- [x] Ferrium Ore income
- [x] Cuprium Ore income
- [x] num forges
- [x] disable purification node
- [x] disable purification building
- [x] outpost savings $ generation rate
- [x] power target
- [x] complexity weight
- [x] delivery box capacity
- [x] delivery num boxes (`--delivery-jobs-per-day`)
