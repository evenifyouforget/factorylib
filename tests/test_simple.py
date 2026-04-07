import numpy as np
import pytest
from numpy import isclose

from factorylib.simple import converger_explicit


def test_always_passes():
    assert True


@pytest.mark.xfail
def test_always_fails():
    assert False


def test_converger_explicit_2in_total_output(in_vec2):
    in_flow = np.array(in_vec2)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


def test_converger_explicit_2in_output_not_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


def test_converger_explicit_2in_input_all_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.any(in_flow < 0.5):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [0.5, 0.5]))


def test_converger_explicit_2in_general(in_vec2):
    in_flow = np.array(in_vec2)
    a, b = in_flow
    out_flow = converger_explicit(in_flow)
    expect_out_flow = np.array([min(a, 1 - min(b, 0.5)), min(b, 1 - min(a, 0.5))])
    assert np.all(isclose(out_flow, expect_out_flow))
