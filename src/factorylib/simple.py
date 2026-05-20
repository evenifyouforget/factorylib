import numpy as np

_EPS = 1e-15


def converger_explicit(in_flow: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Directly calculate the output flows for a converger,
    in a simplified model using continuous steady state
    rather than discrete time stepping. Assumes 1 = full belt.

    Convergers take turns between their inputs using weighted scheduling,
    skipping a turn if the belt in question is unable to supply an item.
    With uniform weights (default), this matches round-robin behavior.

    Args:
        in_flow: Items/second each input belt can supply (real output may be less).
        weights: Fractional bandwidth allocated to each input (must sum to 1).
            None means uniform weights (1/n each).

    Returns:
        Items/second taken from each input, corresponding to the inputs.
        ex. output index 0 corresponds to input index 0.
    """
    if len(in_flow.shape) != 1:
        raise ValueError("Input must be a vector (1D array)")
    n = in_flow.shape[0]
    if n == 0:
        raise ValueError("Degenerate case of 0 inputs is not supported")

    if weights is None:
        weights = np.full(n, 1.0 / n)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,):
            raise ValueError("weights must have same length as in_flow")

    if n == 1:
        return np.minimum(in_flow, weights)  # weights[0] == 1.0 for n=1

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(weights > _EPS, in_flow / weights, np.inf)

    i = int(np.argmin(ratios))
    if ratios[i] >= 1.0:
        return weights.copy()

    # Input i is most undersaturated: give it its full supply.
    a = float(in_flow[i])
    remaining_cap = 1.0 - a
    remaining_in = np.delete(in_flow, i)
    remaining_w = np.delete(weights, i)
    w_sum = float(remaining_w.sum())

    if remaining_cap < _EPS or w_sum < _EPS:
        sub_out = np.zeros(n - 1)
    else:
        remaining_w = remaining_w / w_sum
        sub_out = converger_explicit(remaining_in / remaining_cap, remaining_w) * remaining_cap

    return np.insert(sub_out, i, a)
