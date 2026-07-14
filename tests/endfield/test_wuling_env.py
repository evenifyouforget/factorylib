import numpy as np
import pytest

from factorylib.endfield.wuling import (
    WulingConfig,
    preset_1p2_full,
    preset_1p2e_equiv_1p2d,
    preset_1p2e_full,
    search,
)


def test_replicates_1p2e_full():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_full exactly."""
    result = search(preset_1p2e_full())
    assert result.result.status == "optimal"
    assert result.z == 10
    assert np.allclose(result.metatransfer, [0, 50, 0, 0, 0, 0, 0, 0])
    assert np.allclose(
        result.result.formula_rates,
        [
            8,
            393 / 73,
            59 / 24,
            0,
            13 / 24,
            2,
            11 / 12,
            0,
            1350 / 73,
            0,
            393 / 292,
            410 / 73,
        ],
    )
    assert np.allclose(result.result.resource_slack, [0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(result.result.dollar_output, 206735 / 146)


def test_replicates_1p2e_equiv_1p2d():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_equiv_1p2d."""
    result = search(preset_1p2e_equiv_1p2d())
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 2823 / 2)


def test_replicates_1p2_full():
    """Matches tests/wuling/test_baseline.py::test_wuling_1p2_full (via the
    8-resource collapsed model, purify_node off)."""
    result = search(preset_1p2_full())
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 2229 / 2)


def test_formula_limit_override_ban_ya():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_ya]."""
    config = WulingConfig(formula_limits={"ya": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 205129 / 146)


def test_formula_limit_override_ban_hp():
    """Matches tests/wuling/test_wuling_1p2e.py::test_1p2e_variants[ban_hp]."""
    config = WulingConfig(formula_limits={"hp": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 103335 / 73)


def test_no_purify_building_variant():
    """Matches test_wuling_1p2e.py::test_1p2e_variants[no_purify_building]."""
    config = WulingConfig(formula_limits={"purify": 0})
    result = search(config)
    assert result.result.status == "optimal"
    assert np.isclose(result.result.dollar_output, 119335 / 88)


def test_unknown_formula_limit_raises():
    with pytest.raises(ValueError, match="nonexistent"):
        WulingConfig(formula_limits={"nonexistent": 0})


def test_unknown_formula_output_raises():
    with pytest.raises(ValueError, match="nonexistent"):
        WulingConfig(formula_outputs={"nonexistent": 0})


def test_bad_base_supply_shape_raises():
    with pytest.raises(ValueError, match="base_supply"):
        WulingConfig(base_supply=[0, 1, 2])
