import re
from unittest.mock import patch

import pytest

from factorylib.endfield.cli import main
from factorylib.endfield.refine import RefinedResult


def test_main_default_prints_1p2e_full_dollar(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "206735/146" in out


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
    candidates too, not just leftover base-resource slack."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    delivery_section = out[out.index("Delivery job prediction") :]
    assert "Cuprium Part (sold)" in delivery_section


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
            "0,480,90,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
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
    test_baseline.py::test_wuling_1p2_heavy_xiranite_worth."""
    rc = main(
        [
            "--max-forges",
            "8",
            "--base-supply",
            "0,480,90,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "--no-purify-node",
            "--formula-output",
            "hx_sell=114",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2133/2" in out
    assert "Tied discrete branches" in out
    assert "z=8" in out


def test_main_explicit_purify_building_and_metatransfer_flags(capsys):
    rc = main(
        [
            "--purify-building",
            # 25 Dense Originium Powder
            "--metatransfer",
            "0,0,0,0,0,0,0,0,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            # 25 Ferrium Ore
            "--metatransfer",
            "0,0,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
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
    match = re.search(r"Refined solution: dollar=\S+ \$/min \(([\d.]+) \$/min\)", out)
    assert match, out
    return float(match.group(1))


def test_main_complexity_weight_flag_recovers_more_dollar(capsys):
    """Regression for the demonstrated weight sensitivity: a much lower
    complexity_weight should recover close to LP-optimal $ output."""
    main(["-w", "0.01", "-i", "3000", "-s", "0"])
    relaxed_dollar = _refined_dollar(capsys.readouterr().out)

    main(["-w", "1.0", "-i", "3000", "-s", "0"])
    default_dollar = _refined_dollar(capsys.readouterr().out)

    assert relaxed_dollar >= default_dollar


def test_main_delivery_prediction_prints_percentages(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "selected" in out
    assert "%)" in out


def test_main_diagram_flag_prints_written_path_when_available(capsys):
    with patch(
        "factorylib.endfield.cli.generate_diagram", return_value="/tmp/plan.png"
    ):
        rc = main(["--diagram", "/tmp/plan.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Diagram written to /tmp/plan.png" in out


def test_main_diagram_flag_prints_skip_notice_when_graphviz_unavailable(capsys):
    with patch("factorylib.endfield.cli.generate_diagram", return_value=None):
        rc = main(["--diagram", "/tmp/plan.png"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipped diagram" in out


def test_main_without_diagram_flag_prints_nothing_about_diagrams(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "diagram" not in out.lower()


def test_main_prints_gear_crafting_estimate(capsys):
    """Craft Gear spends *accumulated* Wuling Stock Bill + Components
    (see wuling.py's module docstring for why it's not a Formula) -- the
    CLI reports a separate, out-of-LP days-to-afford estimate instead."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Gear crafting estimate" in out
    assert "Xiranite Component Gear" in out
    assert "Cuprium Component Gear" in out
    assert "Hetonite Component Gear" in out


def test_main_gear_crafting_estimate_shows_never_when_component_rate_is_zero(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    # None of the Components run in the default refined solution (see
    # test_default_gear_priority_penalizes_missing_components's docstring
    # in test_goals.py), so all three gear estimates report "never".
    assert "never at this rate" in out
