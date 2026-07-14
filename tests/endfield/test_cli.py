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
    rc = main(["--limit", "ya=0"])
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
            "0,480,90,180,0,0,0,0,0",
            "--no-purify-node",
            "--formula-output",
            "hp=288",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2229/2" in out
    assert "Tied alternatives" in out


def test_main_no_ties_has_no_alternatives_section(capsys):
    """Also guards against a real regression: sandleaf_powder has zero
    resource cost, so the $-maximizing LP is trivially indifferent to its
    rate (any value up to its limit is equally optimal at $0 marginal
    value) -- a genuine LP degeneracy, but not an economically meaningful
    "tied solution". main() excludes SECONDARY_GOAL_FORMULA_NAMES from
    tie detection for exactly this reason; this test would fail if that
    exclusion were removed."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tied alternatives" not in out
    assert "Tied discrete branches" not in out


def test_main_prints_discrete_branch_ties(capsys):
    """Reproduces the documented hx=$19 z=7-vs-8 discrete tie from
    test_baseline.py::test_wuling_1p2_heavy_xiranite_worth."""
    rc = main(
        [
            "--max-forges",
            "8",
            "--base-supply",
            "0,480,90,180,0,0,0,0,0",
            "--no-purify-node",
            "--formula-output",
            "hx=114",
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
            "--metatransfer",
            "0,0,0,0,0,0,0,0,25",  # 25 Dense Originium Powder
            "--metatransfer",
            "0,0,25,0,0,0,0,0,0",  # 25 Ferrium Ore
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
    rc = main(["-l", "ya=0"])
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
