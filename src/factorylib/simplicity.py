"""Cost of implementing a rate as a physical fraction (splitters/convergers).

Two things make a ratio harder to build in a factory-builder game:
  - a large denominator (deeper splitter trees), and
  - large prime factors in that denominator (uneven, harder splits) --
    2 is the easiest split, 3 is fine, 5 and up gets progressively nastier.
"""

from __future__ import annotations

from fractions import Fraction

_PRIME_WEIGHTS = {2: 1.0, 3: 1.5, 5: 4.0, 7: 6.0}
_LARGE_PRIME_WEIGHT = 10.0  # flat cost per factor of any prime >= 11


def prime_factor_cost(n: int) -> float:
    """Cost of an integer denominator based on its prime factorization.

    Cost accumulates per prime factor (with multiplicity), weighted by how
    hard that prime is to split evenly in-game (see _PRIME_WEIGHTS). Any
    prime factor >= 11 costs a flat _LARGE_PRIME_WEIGHT per occurrence --
    rare enough in practice not to warrant finer-grained tuning.
    """
    n = abs(n)
    if n <= 1:
        return 0.0
    cost = 0.0
    remaining = n
    for prime, weight in _PRIME_WEIGHTS.items():
        while remaining % prime == 0:
            cost += weight
            remaining //= prime
    p = 11
    while p * p <= remaining:
        while remaining % p == 0:
            cost += _LARGE_PRIME_WEIGHT
            remaining //= p
        p += 2
    if remaining > 1:
        cost += _LARGE_PRIME_WEIGHT
    return cost


def fraction_complexity(x: float, max_denom: int = 1000, tol: float = 1e-9) -> float:
    """Cost of implementing rate x as a physical fraction.

    Snaps x to the simplest fraction with denominator <= max_denom and
    prices its denominator via prime_factor_cost. If that fraction isn't
    actually within tol of x, an extra flat penalty is added on top --
    representing that even the best simple approximation available is a
    lossy stand-in for what's really needed (worse than any exactly
    representable denominator within the limit).
    """
    f = Fraction(x).limit_denominator(max_denom)
    cost = prime_factor_cost(f.denominator)
    if abs(float(f) - x) > tol:
        cost += _LARGE_PRIME_WEIGHT
    return cost
