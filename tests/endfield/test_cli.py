import pytest

from factorylib.endfield.cli import main


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


def test_main_prints_alternatives_section_when_tied(capsys):
    rc = main(
        [
            "--max-forges",
            "8",
            "--base-supply",
            "0,480,90,180,0,0,0,0",
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
            "0,480,90,180,0,0,0,0",
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
            "0,50,0,0,0,0,0,0",
            "--metatransfer",
            "0,0,25,0,0,0,0,0",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "206735/146" in out


def test_bad_purify_flags_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["--purify-node", "--no-purify-node"])


def test_unknown_formula_limit_errors():
    with pytest.raises(ValueError, match="nonexistent"):
        main(["--limit", "nonexistent=0"])


def test_bad_kv_format_errors():
    with pytest.raises(SystemExit):
        main(["--limit", "ya"])
