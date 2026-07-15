import numpy as np

from factorylib.alternatives import find_alternatives
from factorylib.breakpoints import max_abs_diff
from factorylib.optimize import Formula


def test_degenerate_tie_surfaces_alternate_vertex():
    # Two formulas compete for the same resource at the same $/unit rate:
    # any split between them is optimal, so the LP's chosen vertex is
    # arbitrary and a small nudge should reveal the other extreme vertex.
    formulas = [
        Formula(consumption=[1.0], output=1.0),
        Formula(consumption=[1.0], output=1.0),
    ]
    result = find_alternatives([10.0], formulas)
    assert np.isclose(result.baseline.dollar_output, 10.0)
    assert len(result.alternatives) >= 1
    for alt in result.alternatives:
        assert np.isclose(alt.dollar_output, 10.0)
        assert max_abs_diff(alt.formula_rates, result.baseline.formula_rates) > 1e-6


def test_non_degenerate_has_no_alternatives():
    formulas = [
        Formula(consumption=[1.0], output=1.0),
        Formula(consumption=[1.0], output=2.0),
    ]
    result = find_alternatives([10.0], formulas)
    assert np.isclose(result.baseline.dollar_output, 20.0)
    assert result.alternatives == []


def test_max_solutions_cap():
    formulas = [Formula(consumption=[1.0], output=1.0) for _ in range(6)]
    result = find_alternatives([10.0], formulas, max_solutions=3)
    assert len(result.alternatives) <= 2


def test_max_solutions_one_skips_perturbation():
    formulas = [
        Formula(consumption=[1.0], output=1.0),
        Formula(consumption=[1.0], output=1.0),
    ]
    result = find_alternatives([10.0], formulas, max_solutions=1)
    assert result.alternatives == []


def test_no_formulas_returns_zero_baseline():
    result = find_alternatives([10.0], [])
    assert result.baseline.status == "zero"
    assert result.alternatives == []


def test_unbounded_baseline_has_no_alternatives():
    # Zero resource consumption + positive output + no limit => unbounded LP.
    formulas = [Formula(consumption=[0.0], output=1.0)]
    result = find_alternatives([10.0], formulas)
    assert result.baseline.status == "unbounded"
    assert result.alternatives == []


def test_every_alternative_matches_baseline_dollar_on_a_real_complex_system():
    """Regression: on the factorylib.endfield Wuling model (48 formulas,
    many zero-$ plumbing steps sharing scarce resources with $-earning
    ones), perturbing a zero-$ formula's output could previously resolve
    to a vertex that was only optimal *for the perturbed problem* --
    e.g. sacrificing most of the real battery-production chain to chase
    an intermediate refining step's own tiny epsilon reward -- and get
    reported as a "tied alternative" despite being far worse under the
    real objective. The old rates-only distinctness check couldn't catch
    this; only checking recomputed dollar_output against baseline can.
    Uses the full, unfiltered formula set (no cli.py-style exclusions)
    to exercise the worst case directly."""
    from factorylib.endfield.wuling import (
        XI_PER_FORGE,
        WulingConfig,
        build_formulas,
        search,
    )

    config = WulingConfig()
    best = search(config)
    formulas = build_formulas(config)
    formulas["hx_make"].limit = config.max_forges - best.z
    supply = config.base_supply + best.z * XI_PER_FORGE + best.metatransfer

    result = find_alternatives(
        supply, list(formulas.values()), epsilon=1e-4, max_solutions=10
    )
    assert np.isclose(result.baseline.dollar_output, 206735 / 146)
    assert len(result.alternatives) >= 1  # the real ya<->jincao_tea tie
    for alt in result.alternatives:
        assert np.isclose(alt.dollar_output, result.baseline.dollar_output)
