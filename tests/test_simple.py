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


def test_weighted_port_matches_duplicated_ports_when_fully_saturated():
    """A port with weight 2/3 competing against a weight-1/3 port should
    behave like that same port duplicated into two separate, equally-
    weighted (1/3 each) ports pulling on the same input value -- *when
    every port is saturated relative to its own weight share* (i.e. each
    port's raw supply is at least its weight, so it's fully bandwidth-
    limited rather than supply-limited). In that regime, weighted
    round-robin literally reduces to "port i gets weights[i] of the
    belt," so splitting one port's weight into N unweighted ports of
    weight 1/N each is exactly the same allocation, just reported as N
    separate numbers instead of one.

    This is NOT true in general (see
    test_weighted_port_does_not_match_duplicated_ports_when_undersaturated)
    -- only in this fully-saturated regime."""
    weights = np.array([2 / 3, 1 / 3])
    for a, b in [(0.7, 0.4), (1.0, 1.0), (5.0, 0.34), (0.6667, 1 / 3)]:
        duplicated = converger_explicit(np.array([a, a, b]))
        weighted = converger_explicit(np.array([a, b]), weights=weights)
        assert isclose(duplicated[0] + duplicated[1], weighted[0])
        assert isclose(duplicated[2], weighted[1])


def test_weighted_port_does_not_match_duplicated_ports_when_undersaturated():
    """Outside full saturation, duplicating a port into two unweighted
    ports is NOT equivalent to giving one port double the weight. At
    a = b = 0.4: each duplicate port's own share is only 1/3, so 0.4
    oversaturates it (every ratio is >= 1, so all three duplicate ports
    are capped at their equal 1/3 share) -- but the single weighted port's
    share is the larger 2/3, which 0.4 does NOT oversaturate (0.4 < 2/3),
    so that port instead gets its full raw supply (0.4) with B (still
    oversaturated at its own 1/3 share) taking the rest. Same total input
    values, different outcome -- a real property of the greedy "give the
    single most-undersaturated port its full supply first" resolution
    order, not a bug, but it does mean weight-N and N duplicate unweighted
    ports are only interchangeable once everything is fully bandwidth-
    saturated (see the test above)."""
    a, b = 0.4, 0.4
    duplicated = converger_explicit(np.array([a, a, b]))
    weighted = converger_explicit(np.array([a, b]), weights=np.array([2 / 3, 1 / 3]))
    assert not isclose(duplicated[0] + duplicated[1], weighted[0])
    assert isclose(duplicated[0] + duplicated[1], 2 / 3)  # capped at 2 * (1/3 share)
    assert isclose(weighted[0], a)  # undersaturated relative to the larger 2/3 share


def test_converger_explicit_in3_cherrypick(in_vec3):
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
    for a_pos in range(3):
        in_flow = np.insert(in_flow_2, a_pos, a)
        expected = np.insert(out_flow_2, a_pos, a)
        assert np.all(isclose(converger_explicit(in_flow), expected))
