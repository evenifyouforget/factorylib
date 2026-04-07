import numpy as np
import pytest
from numpy import isclose

from factorylib.simple import converger_explicit


def test_always_passes():
    assert True


@pytest.mark.xfail
def test_always_fails():
    assert False


def test_converger_explicit_in2_total_output(in_vec2):
    in_flow = np.array(in_vec2)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


def test_converger_explicit_in2_output_not_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


def test_converger_explicit_in2_input_all_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    a = 1/2
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 2))


def test_converger_explicit_in2_general(in_vec2):
    in_flow = np.array(in_vec2)
    a, b = in_flow
    out_flow = converger_explicit(in_flow)
    expect_out_flow = np.array([min(a, 1 - min(b, 0.5)), min(b, 1 - min(a, 0.5))])
    assert np.all(isclose(out_flow, expect_out_flow))

def test_converger_explicit_in3_total_output(in_vec3):
    in_flow = np.array(in_vec3)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))

def test_converger_explicit_in2_output_not_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


def test_converger_explicit_in3_input_all_saturated(in_vec3):
    in_flow = np.array(in_vec3)
    a = 1/3
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 3))

def test_converger_explicit_in3_2_of_3(in_vec2):
    in_flow_2 = np.array(in_vec2)
    a, b = in_flow_2
    out_flow_2 = converger_explicit(in_flow_2)
    ap, bp = out_flow_2
    in_flow = np.array([a, b, 0])
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [ap, bp, 0]))
    in_flow = np.array([a, 0, b])
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [ap, 0, bp]))
    in_flow = np.array([0, a, b])
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [0, ap, bp]))