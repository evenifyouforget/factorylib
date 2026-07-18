import re
from unittest.mock import patch

import pytest

from factorylib.endfield.cli import main
from factorylib.endfield.refine import RefinedResult


@pytest.fixture(autouse=True)
def _diagram_generate_is_noop_by_default(monkeypatch):
    """This test suite itself runs from a real source checkout, so
    _default_diagram_path() resolves to a real output/wuling-diagram.png
    for every main() call below that doesn't pass --diagram/--no-diagram
    -- mock the actual file-writing generate_diagram() call to a no-op
    by default so the other ~40 tests in this file (which don't care
    about diagrams at all) don't have the side effect of writing real
    files into output/. Tests that specifically exercise diagram
    behavior re-patch generate_diagram (and/or _default_diagram_path)
    within their own `with patch(...)` block, which composes fine on
    top of this."""
    monkeypatch.setattr(
        "factorylib.endfield.cli.generate_diagram", lambda *args, **kwargs: None
    )


def test_main_default_prints_1p4_dollar(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "606727/250" in out


def test_main_prints_dollar_for_baseline_and_fitness_for_refined(capsys):
    """The baseline ($-only) section stays a pure $ report (the pp system
    doesn't apply to it -- it never ran any pp-tier formula at all), but
    the refined section -- now scored by pp minus complexity, see
    factorylib.endfield.refine's module docstring -- reports its
    fitness."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.splitlines()
    optimal_idx = next(
        i for i, line in enumerate(lines) if line == "## Optimal solution"
    )
    refined_idx = next(
        i for i, line in enumerate(lines) if line == "### Refined solution"
    )
    optimal_dollar_line = next(
        line for line in lines[optimal_idx:] if line.startswith("dollar = ")
    )
    refined_dollar_line = next(
        line for line in lines[refined_idx:] if line.startswith("dollar = ")
    )
    assert "fitness=" not in optimal_dollar_line
    assert "fitness=" in refined_dollar_line
    assert "Prosperity Points:" in out


def test_main_with_limit_flag(capsys):
    """ya/yc/jincao_tea/jincao_drink are all perfect economic substitutes
    for each other (identical recipe shape/price -- see wuling.py's
    module docstring); 1.4's much larger economy means the $-optimal
    baseline doesn't even happen to run ya/jincao_tea specifically (yc/
    jincao_drink cover that tiny sliver instead), so banning all four is
    needed to see any real $ change at all."""
    rc = main(
        [
            "--limit",
            "ya=0",
            "--limit",
            "yc=0",
            "--limit",
            "jincao_tea=0",
            "--limit",
            "jincao_drink=0",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "605061/250" in out


def test_main_prints_forge_allocation(capsys):
    """1.4 has no z scalar (unlike 1.2e) -- Forge of the Sky's 3-way
    allocation is just ordinary named formulas, already visible in the
    per-formula listing via their own FORMULA_LABELS entry. The default
    power/Water/Acid $ tax (see WulingConfig1p4.power_dollar_tax) now
    consistently prefers the all-Stable-ENV route (1 Carbon/Xiranite)
    over the plain route (2 Stabilized Carbon/Xiranite, itself refined
    from more Carbon) -- previously a pure LP tie between several
    3/9-ish splits."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert (
        "Forge of the Sky, Stable ENV: 1 building → 30/min Xiranite recipe "
        "capacity: [12 = 12.0000] multiples" in out
    )
    assert "Forge of the Sky: 1 building → 30/min Xiranite recipe capacity:" not in out


def test_main_prints_income_breakdown_with_goal_percentage(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "% of 1090 $/min goal" in out
    assert "Income breakdown:" in out
    assert "SC Wuling Battery:" in out
    assert "% of produced" in out
    assert "% of goal" in out


def test_main_income_breakdown_shows_sold_vs_accumulating(capsys):
    """The outpost's $ savings only regenerate at the stock-bill cap, so
    once produced $/min exceeds it, lower-priority goods (see
    wuling.SELL_PRIORITY) accumulate unsold instead of all being sold
    proportionally -- regression for the reported case where Xiranite
    (lowest priority) is produced but the outpost has no savings left to
    buy it, so it just piles up."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "accumulating" in out
    assert "outpost savings only regenerate at 1090 $/min" in out


def test_main_delivery_prediction_includes_unsold_goods(capsys):
    """Goods the outpost can't currently afford to buy (see the income
    breakdown) still physically accumulate, so they must be delivery-job
    candidates too, not just leftover base-resource slack. A near-zero
    stock-bill cap forces this deterministically. Which specific good(s)
    end up "the" unsold one is refine()-seed/pp-tier-weighting sensitive
    (not the thing being tested here), so this only checks that at least
    one *sellable* good (as opposed to a pure delivery-quota/base-resource
    material) shows up as a delivery-job candidate -- i.e. that unsold
    surplus really does feed into the prediction, not just leftover
    slack."""
    rc = main(["--stock-bill-cap", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    delivery_section = out[out.index("Delivery job prediction") :]
    assert "(sellable)" in delivery_section


def test_main_income_breakdown_respects_stock_bill_cap_flag(capsys):
    rc = main(["--stock-bill-cap", "2000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "of 2000 $/min goal" in out


def test_main_prints_material_balance_with_zero_net_for_saturated_resources(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Material balance" in out
    assert "net: [0 = 0.0000]/min" in out  # 1.2e full is fully resource-saturated


def test_main_material_balance_shows_sources_and_sinks(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "from base supply (mining/Forge of the Sky/Metatransfer)" in out
    assert "to Refining Unit: 30/min Cuprium Ore" in out


def test_main_material_balance_lines_show_formula_multiples(capsys):
    """Regression: a source/sink line like "+120/min from Refining Unit:
    30/min Ferrium Ore → 30/min Ferrium" used to read as a mismatch
    (120 vs. the label's own 30/min) without the formula's own multiples
    count -- each line must now append "(N = N.NNNN multiples)" so the
    120/min = 30/min-per-multiple x 4 multiples scaling is explicit."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    balance_section = out[out.index("Material balance") :]
    cuprium_ore_line = next(
        line
        for line in balance_section.splitlines()
        if "to Refining Unit: 30/min Cuprium Ore" in line
    )
    assert re.search(r"\(\S.* multiples\)$", cuprium_ore_line)
    # The base-supply source line has no formula/rate behind it, so it
    # must NOT get a spurious multiples annotation.
    base_supply_line = next(
        line
        for line in balance_section.splitlines()
        if "from base supply (mining/Forge of the Sky/Metatransfer)" in line
    )
    assert "multiples)" not in base_supply_line


def test_main_recipe_listing_does_not_duplicate_income_breakdown_entries(capsys):
    """Sell formulas already itemized in the Income breakdown (name,
    amount, % produced, % goal, sold vs. accumulating) used to also
    appear as a bare "SC Wuling Battery (sellable): [rate] multiples"
    line in the plain recipe listing right above -- strictly less
    detail, so that line must now be skipped whenever the breakdown
    itself is shown."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SC Wuling Battery (sellable):" in out  # still itemized...
    plain_recipe_line = "SC Wuling Battery (sellable): " + re.escape("[")
    assert not re.search(rf"^    {plain_recipe_line}", out, re.MULTILINE)
    # ...but the Material Balance section (more detail than the plain
    # recipe listing, so not considered a duplicate) still shows it.
    assert "to SC Wuling Battery (sellable)" in out


def test_main_delivery_quota_contributors_are_not_hidden_from_delivery_job_prediction(
    capsys,
):
    """Regression for a reported contradiction: materials fully consumed
    by their own delivery_quota_from_*/pp_* bookkeeping formula (e.g.
    Yazhen Powder, Carbon) used to show exactly 0.0 resource_slack even
    while "Delivery quota" reported them as contributing -- hiding them
    from "Delivery job prediction" entirely, which then claimed nothing
    physically accumulates. Every material listed under "Delivery quota"
    must now also appear as a real candidate in the delivery-job
    prediction section (accumulating_display_slack excludes bookkeeping-
    formula consumption so the material's TRUE surplus is used)."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    quota_section = out[
        out.index("Delivery quota:") : out.index("Delivery job prediction")
    ]
    quota_labels = [
        line.split(":")[0].strip()
        for line in quota_section.splitlines()[1:]
        if line.strip().endswith("quota")
    ]
    assert quota_labels  # sanity: the default scenario has quota contributors
    delivery_section = out[out.index("Delivery job prediction") :]
    for label in quota_labels:
        assert label in delivery_section, (
            f"{label} contributes to Delivery quota but never appears in "
            "Delivery job prediction"
        )


def test_main_prints_metatransfer_as_named_item(capsys):
    """The default 1.2e-full metatransfer is 25 Dense Originium Powder;
    the CLI should say so, not print a raw resource-equivalent vector."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "select 25 Dense Originium Powder" in out


def test_main_prints_alternatives_section_when_tied(capsys):
    """The default scenario used to have a genuine tie on its own (several
    Forge-of-the-Sky Carbon-sourcing splits, all equally wasteful of
    otherwise-free Carbon) -- the power/Water/Acid $ tax (see
    WulingConfig1p4.power_dollar_tax) now resolves that tie deliberately,
    so --no-power-dollar-tax restores it here to keep testing the tied-
    alternatives detection mechanism itself, isolated from that concern
    (see test_main_prints_forge_allocation for the tax actually doing its
    job by default)."""
    rc = main(["-n", "8", "--no-power-dollar-tax"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tied alternatives" in out
    assert out.count("606727/250") >= 2  # baseline + at least one alternative


def test_main_prints_discrete_branch_ties(capsys):
    """Forge of the Sky's 3-way allocation is a genuine discrete choice
    (see wuling_1p4.py's module docstring): the default scenario's
    3-plain/9-Stable-ENV split and an all-Stable-ENV-route 12/0 split used
    to be economically tied -- surfaced by the same "Tied alternatives"
    perturbation mechanism used for every other tie, since each
    allocation formula's rate is just another (integer) formula rate to
    it (see wuling.py::search's docstring). The power/Water/Acid $ tax
    (see WulingConfig1p4.power_dollar_tax) now resolves this
    deliberately by default, so --no-power-dollar-tax restores the tie
    to keep testing this specific discrete-branch case. max-solutions is
    raised because, with every tied vertex scoring identically, which
    specific alternatives survive the default truncation depends on
    discovery order (harmless: still discoverable, just not always in a
    short list -- see the completeness note in wuling.py::search's
    docstring)."""
    rc = main(["-n", "8", "--no-power-dollar-tax"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tied alternatives" in out
    assert (
        "Forge of the Sky: 1 building → 30/min Xiranite recipe capacity: "
        "[12 = 12.0000] multiples" in out
    )


def _metatransfer_vector(**amounts: float) -> str:
    """Build a comma-separated len(RESOURCE_NAMES)-value metatransfer
    vector for --metatransfer, keyed by resource name -- avoids hardcoding
    a positional vector tied to a specific RESOURCE_NAMES length/order."""
    from factorylib.endfield.wuling_1p4 import RESOURCE_NAMES

    vec = [0.0] * len(RESOURCE_NAMES)
    for name, amount in amounts.items():
        vec[RESOURCE_NAMES.index(name)] = amount
    return ",".join(str(v) for v in vec)


def test_main_explicit_purify_building_and_metatransfer_flags(capsys):
    rc = main(
        [
            "--purify-building",
            "--metatransfer",
            _metatransfer_vector(dop=25),
            "--metatransfer",
            _metatransfer_vector(ferr=25),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Metatransfer:" in out


def test_main_prints_refined_solution_section(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Most fit solution found" in out
    assert "Refined solution" in out


def test_main_prints_delivery_prediction(capsys):
    """The power/Water/Acid $ tax (see WulingConfig1p4.power_dollar_tax)
    now makes the default refined plan lean enough that nothing
    accumulates unconsumed at all -- this only checks the section
    itself still prints (either outcome, "selected" or "nothing
    accumulates", is a valid report, not a bug)."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Delivery job prediction" in out


def test_main_delivery_flags_accepted(capsys):
    rc = main(
        [
            "--delivery-box-capacity",
            "20000",
            "--delivery-jobs-per-day",
            "3",
            "--delivery-sim-days",
            "10",
            "--delivery-startup-days",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "10 days, 3 jobs/day" in out


def test_main_prints_headroom_warning_when_present(capsys):
    """Forcing headroom_lost via a mock: real scenarios where a move
    happens to fully saturate a previously-slack resource are rare (see
    factorylib.endfield.refine's own tests), so this exercises the CLI's
    print path directly rather than hunting for one."""

    def fake_refine(best_result, best_names, config, goals, search_config, *, backend):
        return RefinedResult(
            rates=best_result.formula_rates,
            pp_output=0.0,
            dollar_output=best_result.dollar_output,
            fitness=0.0,
            formula_names=best_names,
            headroom_lost=["sew"],
        )

    with patch("factorylib.endfield.cli.refine", side_effect=fake_refine):
        rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Warning" in out
    assert "Sewage" in out


def test_main_refine_backend_scipy_runs(capsys):
    rc = main(["--refine-backend", "scipy", "--refine-iterations", "50"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "backend=scipy" in out


def test_main_prints_power_and_delivery_goal_percentage(capsys):
    """Regression: power was never displayed anywhere in the report at
    all (not just "usually 0"), so a real power route (sc_power/
    lc_power) running -- or not -- was invisible either way."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Power:" in out
    assert "% of 7000 W goal" in out
    assert "Delivery quota:" in out
    assert "% of 2 jobs/day goal" in out


def test_main_warns_when_power_goal_unmet(capsys):
    # The search can now genuinely satisfy the *default* 7000 W target
    # (via the SC/LC battery sell-vs-power tradeoff -- see search.py's
    # shift move), so pin an unreachably high target here instead of
    # relying on the default one going unmet, to exercise the shortfall
    # warning path itself regardless of how good the search gets.
    rc = main(["--power-target", "100000000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Warning: only 0.0% of the 100000000 W power goal is met" in out


def test_main_no_power_warning_when_goal_met(capsys):
    def fake_refine(best_result, best_names, config, goals, search_config, *, backend):
        rates = best_result.formula_rates.copy()
        rates[best_names.index("thermal_bank")] = 1000.0  # 50000 W
        return RefinedResult(
            rates=rates,
            pp_output=0.0,
            dollar_output=best_result.dollar_output,
            fitness=0.0,
            formula_names=best_names,
            headroom_lost=[],
        )

    with patch("factorylib.endfield.cli.refine", side_effect=fake_refine):
        rc = main(["--power-target", "7000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "power goal is met" not in out


def test_main_warns_when_stock_bill_goal_unmet(capsys):
    rc = main(["--stock-bill-cap", "100000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stock-bill goal is met" in out


def test_main_no_stock_bill_warning_when_goal_met(capsys):
    rc = main(["--stock-bill-cap", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stock-bill goal is met" not in out


def test_main_warns_when_delivery_goal_unmet(capsys):
    """The $-only baseline's own rates (returned unmodified) never run
    any delivery_quota_from_* formula at all, so delivery quota achieved
    is naturally 0 -- no need to zero anything out explicitly."""

    def fake_refine(best_result, best_names, config, goals, search_config, *, backend):
        return RefinedResult(
            rates=best_result.formula_rates,
            pp_output=0.0,
            dollar_output=best_result.dollar_output,
            fitness=0.0,
            formula_names=best_names,
            headroom_lost=[],
        )

    with patch("factorylib.endfield.cli.refine", side_effect=fake_refine):
        rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "delivery quota goal is met" in out


def test_main_refine_reproducible_with_same_seed(capsys):
    main(["--refine-seed", "3", "--refine-iterations", "200"])
    out1 = capsys.readouterr().out
    main(["--refine-seed", "3", "--refine-iterations", "200"])
    out2 = capsys.readouterr().out
    assert out1 == out2


def test_main_stock_bill_cap_and_power_target_flags_accepted(capsys):
    rc = main(["--stock-bill-cap", "2000", "--power-target", "9000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Most fit solution found" in out


def test_main_random_seed_flag_prints_the_seed_used(capsys):
    rc = main(["-R", "-i", "50"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "using random refine seed" in out
    assert "pass -s" in out


def test_main_short_flags_match_long_flags(capsys):
    rc = main(
        ["-l", "ya=0", "-l", "yc=0", "-l", "jincao_tea=0", "-l", "jincao_drink=0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "605061/250" in out


def test_bad_purify_flags_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["--purify-node", "--no-purify-node"])


def test_unknown_formula_limit_errors():
    with pytest.raises(ValueError, match="nonexistent"):
        main(["--limit", "nonexistent=0"])


def test_bad_kv_format_errors():
    with pytest.raises(SystemExit):
        main(["--limit", "ya"])


def _refined_dollar(out: str) -> float:
    match = re.search(
        r"### Refined solution\n\ndollar = \[\S+ = ([\d.]+)\] \$/min", out
    )
    assert match, out
    return float(match.group(1))


def test_main_complexity_weight_flag_recovers_more_dollar(capsys):
    """Regression for the demonstrated weight sensitivity: a much lower
    complexity_weight should recover close to LP-optimal $ output. Uses a
    wide gap (0.01 vs. 2.0, well past the 0.1 default) since a narrower
    comparison isn't reliably monotonic per-seed once complexity trades
    off against several other goal weights, not just $ alone. Pins
    --power-target low: Planting/Carbon-chain formulas now also draw
    power_flow (see WulingConfig1p4/wuling_1p4.FORMULA_WATTS), making
    Power a genuinely more competitive goal at the margin -- at the
    default power target, a low complexity_weight run correctly
    reallocates SOME resources toward Power's better marginal pp instead
    of squeezing out every last $ (a real pp-optimal tradeoff, not a
    bug), which breaks the clean $-only signal this test wants. A
    trivially-easy power target removes that competition, isolating the
    $-vs-complexity tradeoff this test actually checks."""
    main(["-w", "0.01", "-i", "3000", "-s", "0", "-p", "1"])
    relaxed_dollar = _refined_dollar(capsys.readouterr().out)

    main(["-w", "2.0", "-i", "3000", "-s", "0", "-p", "1"])
    strict_dollar = _refined_dollar(capsys.readouterr().out)

    assert relaxed_dollar >= strict_dollar


def test_main_delivery_prediction_prints_percentages(capsys):
    """The default refined plan is now lean enough (see
    WulingConfig1p4.power_dollar_tax) that nothing accumulates
    unconsumed at all, so nothing would ever get "selected" -- force
    some accumulation the same way test_main_delivery_prediction_
    includes_unsold_goods does (a near-zero stock-bill cap), to exercise
    the percentage-formatting path itself."""
    rc = main(["--stock-bill-cap", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "selected" in out
    assert "%)" in out


def test_main_diagram_flag_prints_written_path_when_rendered(capsys):
    from factorylib.endfield.diagram import DiagramResult

    with patch(
        "factorylib.endfield.cli.generate_diagram",
        return_value=DiagramResult(path="/tmp/plan.png", rendered=True),
    ):
        rc = main(["--diagram", "/tmp/plan.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Diagram written to /tmp/plan.png" in out


def test_main_diagram_flag_prints_dot_fallback_notice_when_not_rendered(capsys):
    """When graphviz is installed but its system `dot` executable isn't,
    generate_diagram falls back to writing raw .dot source instead of an
    image -- the CLI must say so, not claim a rendered diagram exists."""
    from factorylib.endfield.diagram import DiagramResult

    with patch(
        "factorylib.endfield.cli.generate_diagram",
        return_value=DiagramResult(path="/tmp/plan.dot", rendered=False),
    ):
        rc = main(["--diagram", "/tmp/plan.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/tmp/plan.dot" in out
    assert "dot` executable isn't" in out


def test_main_diagram_flag_prints_skip_notice_when_graphviz_unavailable(capsys):
    with patch("factorylib.endfield.cli.generate_diagram", return_value=None):
        rc = main(["--diagram", "/tmp/plan.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipped diagram" in out


def test_main_no_diagram_flag_prints_nothing_about_diagrams(capsys):
    """--no-diagram must suppress even the source-checkout default path
    (this test suite itself runs from a source checkout, so without
    --no-diagram the default would kick in and actually try to render)."""
    rc = main(["--no-diagram"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "diagram" not in out.lower()


def test_main_no_diagram_flag_overrides_explicit_diagram_path(capsys):
    with patch("factorylib.endfield.cli.generate_diagram") as mock_generate:
        rc = main(["--diagram", "/tmp/plan.png", "--no-diagram"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "diagram" not in out.lower()
    mock_generate.assert_not_called()


def test_main_uses_default_diagram_path_when_neither_flag_given(capsys):
    """Without --diagram or --no-diagram, main() should fall back to
    _default_diagram_path() (mocked here to a fake path so this doesn't
    depend on whether this environment is actually a source checkout)."""
    from factorylib.endfield.diagram import DiagramResult

    with (
        patch(
            "factorylib.endfield.cli._default_diagram_path",
            return_value="/tmp/fake_default/plan.png",
        ),
        patch(
            "factorylib.endfield.cli.generate_diagram",
            return_value=DiagramResult(
                path="/tmp/fake_default/plan.png", rendered=True
            ),
        ) as mock_generate,
    ):
        rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Diagram written to /tmp/fake_default/plan.png" in out
    assert mock_generate.call_args.args[-1] == "/tmp/fake_default/plan.png"


def test_main_explicit_diagram_path_overrides_default(capsys):
    from factorylib.endfield.diagram import DiagramResult

    with (
        patch(
            "factorylib.endfield.cli._default_diagram_path",
            return_value="/tmp/fake_default/plan.png",
        ),
        patch(
            "factorylib.endfield.cli.generate_diagram",
            return_value=DiagramResult(path="/tmp/explicit.png", rendered=True),
        ) as mock_generate,
    ):
        rc = main(["--diagram", "/tmp/explicit.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Diagram written to /tmp/explicit.png" in out
    assert mock_generate.call_args.args[-1] == "/tmp/explicit.png"


def test_default_diagram_path_is_none_outside_a_source_checkout():
    from factorylib.endfield.cli import _default_diagram_path

    with patch("os.path.isdir", return_value=False):
        assert _default_diagram_path() is None


def test_default_diagram_path_is_output_dir_inside_a_source_checkout():
    """This test suite itself runs from a real source checkout (this
    repo has a .git directory), so the real (unmocked) function should
    resolve to the real default."""
    from factorylib.endfield.cli import _DEFAULT_DIAGRAM_PATH, _default_diagram_path

    assert _default_diagram_path() == _DEFAULT_DIAGRAM_PATH
