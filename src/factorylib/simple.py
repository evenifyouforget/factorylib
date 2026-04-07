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
    if in_flow.shape != (2,):
        raise ValueError("Cases other than 2 inputs not supported")
    a = in_flow[0]
    b = in_flow[1]
    return np.array([min(a, 1 - min(b, 0.5)), min(b, 1 - min(a, 0.5))])
