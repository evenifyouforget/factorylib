"""Golden regression + CLI smoke tests for factorylib.endfield.main.

main.py is the design oracle: its ~180 Recipe declarations encode the full
Endfield 1.4 production model (buildings, sell prices, the Prosperity
Points mixed-objective overlay). These tests don't re-derive the "correct"
answer independently -- they pin the actual $-optimal objective value this
reconciled code produces, so any future change to the model or the solver
plumbing that shifts the answer gets caught.

The golden figures below were captured by actually running the reconciled
CLI (not picked from memory or the pre-refactor code) and were also
cross-checked byte-for-byte against the pre-refactor main.py's stdout
report during the library reconciliation.
"""

from __future__ import annotations

import re

import pytest

from factorylib.endfield.main import main

SELLABLE_GOLDEN_OBJECTIVE = 2293.8239875000017
MIXED_GOLDEN_OBJECTIVE = 18748.097174815644

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
