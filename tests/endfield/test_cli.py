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


def test_main_default_prints_1p2e_full_dollar(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "206735/146" in out


def test_main_prints_dollar_for_baseline_and_fitness_for_refined(capsys):
    """The baseline ($-only) section stays a pure $ report (the pp system
    doesn't apply to it -- it never ran any pp-tier formula at all), but
    the refined section -- now scored by pp minus complexity, see
    factorylib.endfield.refine's module docstring -- reports its
    fitness."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    optimal_line = next(
        line for line in out.splitlines() if line.startswith("Optimal solution:")
    )
    refined_line = next(
        line for line in out.splitlines() if "Refined solution:" in line
    )
    assert "fitness=" not in optimal_line
    assert "fitness=" in refined_line
    assert "Prosperity Points:" in out


def test_main_with_limit_flag(capsys):
    """jincao_tea is a perfect economic substitute for ya (identical
    recipe shape/price -- see wuling.py's module docstring), so banning
    ya alone must also ban jincao_tea to reproduce the historical
    ban_ya figure."""
    rc = main(["--limit", "ya=0", "--limit", "jincao_tea=0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "205129/146" in out


def test_main_prints_forge_allocation(capsys):
    """Spells out what z means: how many forges feed Xiranite supply vs.
    cap Heavy Xiranite, instead of a bare "z=10"."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "10 -> Xiranite supply" in out
    assert "2 -> Heavy Xiranite capacity" in out


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
    assert "net: 0/min" in out  # 1.2e full is fully resource-saturated


def test_main_material_balance_shows_sources_and_sinks(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "from base supply (mining/Forge of the Sky/Metatransfer)" in out
    assert "to Cuprium Ore Refining" in out


def test_main_prints_metatransfer_as_named_item(capsys):
    """The default 1.2e-full metatransfer is 25 Dense Originium Powder;
    the CLI should say so, not print a raw resource-equivalent vector."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "select 25 Dense Originium Powder" in out


def test_main_prints_alternatives_section_when_tied(capsys):
    rc = main(
        [
            "--max-forges",
            "8",
            "--base-supply",
            "0,480,90,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "--no-purify-node",
            "--formula-output",
            "hp_sell=288",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2229/2" in out
    assert "Tied alternatives" in out


def test_main_no_ties_when_jincao_substitute_is_banned(capsys):
    """Guards against a real regression: sandleaf_powder has zero
    resource cost, so the $-maximizing LP is trivially indifferent to its
    rate (any value up to its limit is equally optimal at $0 marginal
    value) -- a genuine LP degeneracy, but not an economically meaningful
    "tied solution". main() excludes SECONDARY_GOAL_FORMULA_NAMES (and
    the plumbing that solely feeds them) from tie detection for exactly
    this reason; this test would fail if that exclusion were removed.
    jincao_tea is banned here to isolate this from the separate, genuine
    ya<->jincao_tea tie the default scenario now has (see the test
    below)."""
    rc = main(["--limit", "jincao_tea=0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tied alternatives" not in out
    assert "Tied discrete branches" not in out


def test_main_default_scenario_shows_genuine_ya_jincao_tea_tie(capsys):
    """jincao_tea is a perfect economic substitute for ya (identical
    recipe shape and price -- see wuling.py's module docstring), so the
    default 1.2e-full scenario now has a real tied alternative: this is
    exactly the kind of "genuine choice between two strategies" tie
    detection is meant to surface, unlike the zero-$ secondary-goal
    degeneracy the test above excludes."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tied alternatives" in out
    assert "206735/146" in out  # alternative's recomputed $ matches baseline


def test_main_prints_discrete_branch_ties(capsys):
    """Reproduces the documented hx=$19 z=7-vs-8 discrete tie from
    test_baseline.py::test_wuling_1p2_heavy_xiranite_worth -- now surfaced
    by the same "Tied alternatives" perturbation mechanism used for every
    other tie, since z is just another (integer) formula rate to it (see
    wuling.py::search's docstring). max-solutions is raised because, with
    every tied vertex scoring identically, which specific alternatives
    survive the default truncation depends on discovery order -- this
    specific z=7 branch isn't guaranteed to be among the first few found
    otherwise (harmless: it's still discoverable, just not always in a
    short list -- see the completeness note in wuling.py::search's
    docstring)."""
    rc = main(
        [
            "--max-forges",
            "8",
            "--base-supply",
            "0,480,90,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "--no-purify-node",
            "--formula-output",
            "hx_sell=114",
            "--max-solutions",
            "8",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2133/2" in out
    assert "Tied alternatives" in out
    assert "Forge of the Sky (→ Heavy Xiranite capacity): [1 = 1.0000] multiples" in out


def test_main_explicit_purify_building_and_metatransfer_flags(capsys):
    rc = main(
        [
            "--purify-building",
            # 25 Dense Originium Powder
            "--metatransfer",
            "0,0,0,0,0,0,0,0,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            # 25 Ferrium Ore
            "--metatransfer",
            "0,0,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "206735/146" in out


def test_main_prints_refined_solution_section(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Most fit solution found" in out
    assert "Refined solution" in out


def test_main_prints_delivery_prediction(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Delivery job prediction" in out
    assert "Sandleaf Powder: selected" in out


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

    def fake_refine(base, config, goals, search_config, *, backend):
        return RefinedResult(
            rates=base.result.formula_rates,
            pp_output=0.0,
            dollar_output=base.result.dollar_output,
            fitness=0.0,
            formula_names=base.formula_names,
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
    def fake_refine(base, config, goals, search_config, *, backend):
        rates = base.result.formula_rates.copy()
        rates[base.formula_names.index("thermal_bank")] = 1000.0  # 50000 W
        return RefinedResult(
            rates=rates,
            pp_output=0.0,
            dollar_output=base.result.dollar_output,
            fitness=0.0,
            formula_names=base.formula_names,
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

    def fake_refine(base, config, goals, search_config, *, backend):
        return RefinedResult(
            rates=base.result.formula_rates,
            pp_output=0.0,
            dollar_output=base.result.dollar_output,
            fitness=0.0,
            formula_names=base.formula_names,
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
    rc = main(["-l", "ya=0", "-l", "jincao_tea=0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "205129/146" in out


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
    match = re.search(r"Refined solution: dollar=\[\S+ = ([\d.]+)\] \$/min", out)
    assert match, out
    return float(match.group(1))


def test_main_complexity_weight_flag_recovers_more_dollar(capsys):
    """Regression for the demonstrated weight sensitivity: a much lower
    complexity_weight should recover close to LP-optimal $ output. Uses a
    wide gap (0.01 vs. 2.0, well past the 0.1 default) since a narrower
    comparison isn't reliably monotonic per-seed once complexity trades
    off against several other goal weights, not just $ alone."""
    main(["-w", "0.01", "-i", "3000", "-s", "0"])
    relaxed_dollar = _refined_dollar(capsys.readouterr().out)

    main(["-w", "2.0", "-i", "3000", "-s", "0"])
    strict_dollar = _refined_dollar(capsys.readouterr().out)

    assert relaxed_dollar >= strict_dollar


def test_main_delivery_prediction_prints_percentages(capsys):
    rc = main([])
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
