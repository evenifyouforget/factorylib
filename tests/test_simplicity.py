from factorylib.simplicity import fraction_complexity, prime_factor_cost


def test_prime_factor_cost_trivial():
    assert prime_factor_cost(1) == 0.0
    assert prime_factor_cost(0) == 0.0


def test_prime_factor_cost_orders_small_primes_by_difficulty():
    assert prime_factor_cost(2) < prime_factor_cost(3)
    assert prime_factor_cost(3) < prime_factor_cost(5)
    assert prime_factor_cost(5) < prime_factor_cost(7)
    assert prime_factor_cost(7) < prime_factor_cost(11)


def test_prime_factor_cost_multiplicity():
    # 8 = 2^3: three times the cost of a single factor of 2.
    assert prime_factor_cost(8) == 3 * prime_factor_cost(2)
    # 1024 = 2^10.
    assert prime_factor_cost(1024) == 10 * prime_factor_cost(2)


def test_prime_factor_cost_composite():
    # 96 = 2^5 * 3
    assert prime_factor_cost(96) == 5 * prime_factor_cost(2) + prime_factor_cost(3)


def test_prime_factor_cost_large_prime():
    # Primes >= 11 all cost the same flat rate (no finer-grained tuning).
    assert prime_factor_cost(11) == prime_factor_cost(13) == prime_factor_cost(101)
    assert prime_factor_cost(11) > prime_factor_cost(7)


def test_prime_factor_cost_composite_of_large_primes():
    # 143 = 11 * 13: two large-prime factors, found via trial division.
    assert prime_factor_cost(143) == 2 * prime_factor_cost(11)


def test_fraction_complexity_zero_and_integers_are_free():
    assert fraction_complexity(0.0) == 0.0
    assert fraction_complexity(3.0) == 0.0


def test_fraction_complexity_matches_prime_factor_cost_of_denominator():
    assert fraction_complexity(0.5) == prime_factor_cost(2)
    assert fraction_complexity(19 / 96) == prime_factor_cost(96)


def test_fraction_complexity_harder_fraction_costs_more():
    # 19/96 (a documented "hard to build" case) should cost more than 1/2.
    assert fraction_complexity(19 / 96) > fraction_complexity(0.5)


def test_fraction_complexity_unrepresentable_gets_extra_penalty():
    import math

    exact = fraction_complexity(1 / 3)
    unrepresentable = fraction_complexity(math.pi, max_denom=10)
    assert unrepresentable > exact
