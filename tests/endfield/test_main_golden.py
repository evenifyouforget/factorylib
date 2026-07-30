"""Golden regression + CLI smoke tests for factorylib.endfield.main.

main.py is the design oracle: its ~180 Recipe declarations encode the
full Endfield 1.4 production model (buildings, sell prices, the
Prosperity Points mixed-objective overlay). These tests don't re-derive
the "correct" answer independently. Instead, they pin the actual
$-optimal objective value this reconciled code produces, so any future
change to the model or the solver plumbing that shifts the answer gets
caught.

The golden figures below were captured by actually running the
reconciled CLI (not picked from memory or the pre-refactor code), and
were also cross-checked byte-for-byte against the pre-refactor main.py's
stdout report during the library reconciliation.

MIXED_GOLDEN_OBJECTIVE was recaptured at 18749.06959502713 (previously
18748.097174815644) after fixing factorylib.optimize.solve() to pass
mip_rel_gap=1e-9 to scipy's milp(). The "mixed" target's many integer
Award-Points-tier recipes gave HiGHS's default relative gap (~1e-4)
enough room to stop at a solution merely close to optimal rather than
the true one -- confirmed by reproducing the same old, short-by-~0.97
figure even from the pre-fix *unmodified* model with a manually tightened
gap, and confirmed as the true optimum by getting the same 18749.0696...
figure regardless of whether an economically-irrelevant extra recipe
(the Test Area Purification Node, added by a later commit) is present.
"""

from __future__ import annotations

import re

import pytest

from factorylib.endfield.main import main

SELLABLE_GOLDEN_OBJECTIVE = 2293.8239875000017
MIXED_GOLDEN_OBJECTIVE = 18749.06959502713

_SCORE_RE = re.compile(r"Maximized score: (\S+)")


def _run_main(argv, capsys):
    import sys

    old_argv = sys.argv
    sys.argv = ["factorylib-endfield", *argv]
    try:
        main()
    finally:
        sys.argv = old_argv
    return capsys.readouterr().out


def _extract_score(output: str) -> float:
    match = _SCORE_RE.search(output)
    assert match is not None, "report did not contain a 'Maximized score' line"
    return float(match.group(1))


def test_sellable_target_golden_objective(capsys):
    output = _run_main(["-t", "sellable"], capsys)
    assert "# Result 0:" in output
    assert _extract_score(output) == pytest.approx(SELLABLE_GOLDEN_OBJECTIVE)


def test_default_target_matches_sellable(capsys):
    """--target isn't required by argparse; omitting it takes the same
    non-'mixed' code path as -t sellable."""
    output = _run_main([], capsys)
    assert _extract_score(output) == pytest.approx(SELLABLE_GOLDEN_OBJECTIVE)


def test_mixed_target_golden_objective(capsys):
    output = _run_main(["-t", "mixed"], capsys)
    assert "# Result 0:" in output
    assert _extract_score(output) == pytest.approx(MIXED_GOLDEN_OBJECTIVE)


def test_force_fractions_flag_runs_and_reports_fractions(capsys):
    output = _run_main(["-t", "sellable", "-f"], capsys)
    assert "# Result 0:" in output
    # Forced-fraction mode prints Fraction2's "[n/d = float]" rendering for
    # every nonzero recipe multiple.
    assert "[1 = 1.0] multiples of Starting Materials" in output


def test_graph_outfile_renders_file(tmp_path, capsys):
    outfile = tmp_path / "graph"
    _run_main(["-t", "sellable", "-o", str(outfile)], capsys)
    assert outfile.exists()
    assert (tmp_path / "graph.pdf").exists()


def test_xiragen_amount_is_configurable_without_editing_source(capsys):
    """--xiragen used to require commenting/uncommenting a line in
    Starting Materials by hand. Setting it to 0 reproduces the historical
    pre-Xiragen-restoration figure exactly."""
    output = _run_main(["-t", "sellable", "--xiragen", "0"], capsys)
    assert _extract_score(output) == pytest.approx(1921.663987500002)


def test_ore_and_inergen_supply_are_configurable(capsys):
    """Every Starting Materials amount is a CLI argument now, not a
    hardcoded literal -- zeroing them all out still solves (Metatransfer/
    Forge of the Sky allocations are independent of raw ore supply) but
    for a much smaller objective than the default supply."""
    output = _run_main(
        [
            "-t",
            "sellable",
            "--originium-ore",
            "0",
            "--ferrium-ore",
            "0",
            "--cuprium-ore",
            "0",
            "--inergen",
            "0",
            "--xiragen",
            "0",
        ],
        capsys,
    )
    assert "# Result 0:" in output
    assert _extract_score(output) == pytest.approx(188.02083333333346)


def test_test_area_purification_node_disabled_by_default(capsys):
    """The recipe is always present now (so its Allocation material still
    shows up, at net 0, in the material balance sheet), but must never
    actually run: 0 max_multiples by default."""
    output = _run_main(["-t", "sellable"], capsys)
    assert "multiples of Test Area Purification Node" not in output


def test_test_area_max_multiples_enables_a_strictly_additional_option(capsys):
    """Enabling the Test Area Purification Node used to require
    uncommenting a line by hand. It's a real, additional production
    option -- enabling it can only help or leave the $-optimum unchanged,
    never hurt."""
    baseline = _extract_score(_run_main(["-t", "sellable"], capsys))
    enabled_output = _run_main(
        ["-t", "sellable", "--test-area-max-multiples", "12"], capsys
    )
    assert "multiples of Test Area Purification Node" in enabled_output
    assert _extract_score(enabled_output) >= baseline


def test_jincao_disabled_by_default(capsys):
    """The Jincao production chain (planting, seed-picking, bottling) is
    always present in the model -- so "Jincao Drink"/"Jincao Tea" as bare
    material names still show up (at net 0) in the balance sheet -- but
    the Sell and Jincao->Carbon refine recipes that would make any of it
    worthwhile are absent unless explicitly enabled."""
    output = _run_main(["-t", "sellable"], capsys)
    assert "multiples of Sell (0 W): Jincao Drink" not in output
    assert "multiples of Sell (0 W): Jincao Tea" not in output
    assert "multiples of Refining Unit (5 W): 30/min Jincao -->" not in output


def test_enable_jincao_ties_with_yazhen_rather_than_improving_on_it(capsys):
    """Jincao Drink/Tea sell for the same $ as Yazhen Syringe C/A (16 and
    22 respectively) via the same Filling/Packaging Unit capacity, so
    enabling Jincao doesn't change the $-optimum at all -- it only gives
    the solver more (economically identical, harder-to-build) ways to
    reach the same number."""
    baseline = _extract_score(_run_main(["-t", "sellable"], capsys))
    with_jincao = _extract_score(
        _run_main(["-t", "sellable", "--enable-jincao"], capsys)
    )
    assert with_jincao == pytest.approx(baseline)
