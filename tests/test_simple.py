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
    a = 1 / 2
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 2))


def test_converger_explicit_in2_general(in_vec2):
    in_flow = np.array(in_vec2)
    a, b = in_flow
    out_flow = converger_explicit(in_flow)
    # Handwritten explicit formula that is already validated
    expect_out_flow = np.array([min(a, 1 - min(b, 0.5)), min(b, 1 - min(a, 0.5))])
    assert np.all(isclose(out_flow, expect_out_flow))


def test_converger_explicit_in3_total_output(in_vec3):
    in_flow = np.array(in_vec3)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


def test_converger_explicit_in3_output_not_saturated(in_vec3):
    in_flow = np.array(in_vec3)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


def test_converger_explicit_in3_input_all_saturated(in_vec3):
    in_flow = np.array(in_vec3)
    a = 1 / 3
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 3))


def test_converger_explicit_in3_2_of_3(in_vec2):
    in_flow_2 = np.array(in_vec2)
    out_flow_2 = converger_explicit(in_flow_2)
    for zero_pos in range(3):
        in_flow = np.insert(in_flow_2, zero_pos, 0)
        expected = np.insert(out_flow_2, zero_pos, 0)
        assert np.all(isclose(converger_explicit(in_flow), expected))


def test_converger_explicit_in3_cherrypick(in_vec3):
    a, b, c = in_vec3
    if not 0 < a < min([1 / 3, b, c]):
        pytest.skip("Test does not cover this range")
    # a always gets its full share, so b and c must fight for the remaining portion (1 - a)
    # this reduces to a 2-input subproblem
    # since the solver is normalized to 1 item/s, we need to scale
    # b, c -> 1 - a
    # up to
    # b / (1 - a), c / (1 - a) -> 1
    subproblem_scale = 1 - a
    subproblem_in_flow_2 = np.array([b, c]) / subproblem_scale
    in_flow_2 = np.array([b, c])
    out_flow_2 = converger_explicit(subproblem_in_flow_2) * subproblem_scale
    for a_pos in range(3):
        in_flow = np.insert(in_flow_2, a_pos, a)
        expected = np.insert(out_flow_2, a_pos, a)
        assert np.all(isclose(converger_explicit(in_flow), expected))
