import numpy as np
import pytest
from numpy import isclose

from factorylib.simple import converger_explicit

IN2_TUPLES = [(i / 10, j / 10) for i in range(11) for j in range(11)]
IN3_TUPLES = [
    (i / 10, j / 10, k / 10) for i in range(11) for j in range(11) for k in range(11)
]


def test_always_passes():
    assert True


@pytest.mark.xfail
def test_always_fails():
    assert False


@pytest.mark.parametrize("in_vec2", IN2_TUPLES)
def test_converger_explicit_in2_total_output(in_vec2):
    in_flow = np.array(in_vec2)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


@pytest.mark.parametrize("in_vec2", IN2_TUPLES)
def test_converger_explicit_in2_output_not_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


@pytest.mark.parametrize("in_vec2", IN2_TUPLES)
def test_converger_explicit_in2_input_all_saturated(in_vec2):
    in_flow = np.array(in_vec2)
    a = 1 / 2
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 2))


@pytest.mark.parametrize("in_vec2", IN2_TUPLES)
def test_converger_explicit_in2_general(in_vec2):
    in_flow = np.array(in_vec2)
    a, b = in_flow
    out_flow = converger_explicit(in_flow)
    # Handwritten explicit formula that is already validated
    expect_out_flow = np.array([min(a, 1 - min(b, 0.5)), min(b, 1 - min(a, 0.5))])
    assert np.all(isclose(out_flow, expect_out_flow))


@pytest.mark.parametrize("in_vec3", IN3_TUPLES)
def test_converger_explicit_in3_total_output(in_vec3):
    in_flow = np.array(in_vec3)
    out_flow = converger_explicit(in_flow)
    assert isclose(np.sum(out_flow), min(np.sum(in_flow), 1))


@pytest.mark.parametrize("in_vec3", IN3_TUPLES)
def test_converger_explicit_in3_output_not_saturated(in_vec3):
    in_flow = np.array(in_vec3)
    if np.sum(in_flow) > 1:
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, in_flow))


@pytest.mark.parametrize("in_vec3", IN3_TUPLES)
def test_converger_explicit_in3_input_all_saturated(in_vec3):
    in_flow = np.array(in_vec3)
    a = 1 / 3
    if np.any(in_flow < a):
        pytest.skip("Test does not cover this range")
    out_flow = converger_explicit(in_flow)
    assert np.all(isclose(out_flow, [a] * 3))


@pytest.mark.parametrize("zero_pos", range(3))
@pytest.mark.parametrize("in_vec2", IN2_TUPLES)
def test_converger_explicit_in3_2_of_3(in_vec2, zero_pos):
    in_flow_2 = np.array(in_vec2)
    out_flow_2 = converger_explicit(in_flow_2)
    in_flow = np.insert(in_flow_2, zero_pos, 0)
    expected = np.insert(out_flow_2, zero_pos, 0)
    assert np.all(isclose(converger_explicit(in_flow), expected))


def test_converger_explicit_2d_raises():
    with pytest.raises(ValueError, match="1D array"):
        converger_explicit(np.array([[0.3, 0.7]]))


def test_converger_explicit_empty_raises():
    with pytest.raises(ValueError, match="0 inputs"):
        converger_explicit(np.array([]))


def test_converger_explicit_weights_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        converger_explicit(np.array([0.5, 0.5]), weights=np.array([0.5]))


def test_converger_explicit_zero_weight():
    # weights=[1, 0]: port 1 gets no bandwidth; hits the w_sum<_EPS path
    out = converger_explicit(np.array([0.8, 0.6]), weights=np.array([1.0, 0.0]))
    assert np.allclose(out, [0.8, 0.0])


@pytest.mark.parametrize(
    "a, b", [(0.7, 0.4), (1.0, 1.0), (5.0, 0.34), (0.6667, 1 / 3), (0.0, 0.5)]
)
def test_duplicated_port_matches_a_merged_double_weight_double_supply_port(a, b):
    """Splitting a resource across two equally-weighted ports is
    equivalent to merging it into one double-weighted port fed the
    *combined* supply of both: converge([A, A, B]) (2 ports of A, each
    weight 1/3, i.e. up to 2*A of resource A available across both
    slots) == converge([2*A, B], weights=[2/3, 1/3]) (1 port of 2*A,
    weight 2/3). This holds unconditionally, for any A/B, not just in
    some saturated regime -- an earlier version of this test compared
    converge([A, A, B]) against converge([A, B], weights=[2/3, 1/3]),
    which mismatches the total A supply available (A on one side, 2*A
    on the other) and so only appeared to match in the specific case
    where both sides happened to be fully bandwidth-saturated anyway.
    Once the supply is scaled to match the weight, the two really are
    the same allocation in general, just reported as 2 numbers vs. 1."""
    # weights must be fractional bandwidth (sum to 1) per converger_explicit's
    # own contract -- passing raw un-normalized counts like [2, 1] directly
    # is NOT equivalent and gives a different (wrong) answer.
    weights = np.array([2 / 3, 1 / 3])
    duplicated = converger_explicit(np.array([a, a, b]))
    merged = converger_explicit(np.array([2 * a, b]), weights=weights)
    assert isclose(duplicated[0] + duplicated[1], merged[0])
    assert isclose(duplicated[2], merged[1])


@pytest.mark.parametrize("a_pos", range(3))
@pytest.mark.parametrize("in_vec3", IN3_TUPLES)
def test_converger_explicit_in3_cherrypick(in_vec3, a_pos):
    a, b, c = in_vec3
    if not 0 < a < min([1 / 3, b, c]):
        pytest.skip("Test does not cover this range")
    # a always gets its full share, so b and c must
    # compete for the remaining portion (1 - a)
    # this reduces to a 2-input subproblem
    # since the solver is normalized to 1 item/s, we need to scale
    # b, c -> 1 - a
    # up to
    # b / (1 - a), c / (1 - a) -> 1
    subproblem_scale = 1 - a
    subproblem_in_flow_2 = np.array([b, c]) / subproblem_scale
    in_flow_2 = np.array([b, c])
    out_flow_2 = converger_explicit(subproblem_in_flow_2) * subproblem_scale
    in_flow = np.insert(in_flow_2, a_pos, a)
    expected = np.insert(out_flow_2, a_pos, a)
    assert np.all(isclose(converger_explicit(in_flow), expected))
