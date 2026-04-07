import numpy as np
import pytest
from numpy import isclose

from factorylib.simple import converger_explicit


def test_always_passes():
    assert True


@pytest.mark.xfail
def test_always_fails():
    assert False


def test_converger_explicit_saturation(in_vec2):
    in_flow = np.array(in_vec2)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


def test_converger_explicit_output_not_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


def test_converger_explicit_input_all_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.any(in_flow < 0.5):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [0.5, 0.5]))
