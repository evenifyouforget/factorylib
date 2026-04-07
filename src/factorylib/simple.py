import numpy as np


def converger_explicit(in_flow: np.ndarray) -> np.ndarray:
    """
    Directly calculate the output flows for a converger,
    in a simplified model using continuous steady state
    rather than discrete time stepping. Assumes 1 = full belt.

    Convergers take turns between their inputs, skipping a turn if
    the belt in question is unable to supply an item at that time.

    Args:
        in_flow (np.ndarray): Items/second that each input belt
        is able to supply (real output may be less than this)

    Raises:
        ValueError: Not yet supported cases

    Returns:
        np.ndarray: Items/second of each item, corresponding to the inputs
        ex. output index 0 corresponds to input index 0
    """
    if len(in_flow.shape) != 1:
        raise ValueError("Input must be a vector (1D array)")
    n = in_flow.shape[0]
    if n == 0:
        raise ValueError("Degenerate case of 0 inputs is not supported")
    if n == 1:
        return np.minimum(in_flow, np.array([1.0]))
    i = np.argmin(in_flow)
    a = in_flow[i]
    if a >= 1 / n:
        # All inputs are saturated, so we can just split the output evenly
        return np.full_like(in_flow, 1 / n)
    # Small input will always get its turn and output the full value
    # We can remove it and recursively solve the subproblem
    subproblem_scale = 1 - a
    subproblem_in_flow = np.delete(in_flow, i) / subproblem_scale
    subproblem_out_flow = converger_explicit(subproblem_in_flow) * subproblem_scale
    out_flow = np.insert(subproblem_out_flow, i, a)
    return out_flow
