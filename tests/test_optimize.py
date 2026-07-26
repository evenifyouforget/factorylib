"""Tests for factorylib.optimize.solve()/OptimizeResult/material_balance().

This replaces the old test_optimize.py, which tested the now-deleted plain-LP
Formula/maximize_dollar API (no recipe algebra, no MILP). The new API works
directly over Material/Recipe expressions, matching factorylib.endfield.main
(the design oracle this was ported from).
"""

from __future__ import annotations

import numpy as np
import pytest

from factorylib.material import GAS, SOLID, Material, Recipe
from factorylib.optimize import material_balance, solve


def test_single_recipe_supply_binding():
    """Supply constraint is tighter than the unbounded recipe's own limit."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    result = solve(set(), [supply, sell], dollar)
    assert result.status == 0
    assert np.isclose(result.objective, 15.0)
    assert np.allclose(result.multiples, [1.0, 5.0])


def test_recipe_max_multiples_binding():
    """The recipe's own max_multiples is the binding constraint, not supply."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(100 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 4 * dollar, name="Sell", max_multiples=1.0)
    result = solve(set(), [supply, sell], dollar)
    assert result.status == 0
    assert np.isclose(result.objective, 4.0)
    assert np.isclose(result.multiples[1], 1.0)


def test_competing_recipes_prefer_better_dollar_rate():
    """Two recipes compete for one resource; the solver should fully prefer
    the higher $/unit recipe over the worse one."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(6 * ore, name="Supply", max_multiples=1)
    worse = Recipe(-1 * ore + 1 * dollar, name="Worse Sell")
    better = Recipe(-1 * ore + 2 * dollar, name="Better Sell")
    result = solve(set(), [supply, worse, better], dollar)
    assert result.status == 0
    assert np.isclose(result.objective, 12.0)
    assert np.isclose(result.multiples[1], 0.0, atol=1e-9)
    assert np.isclose(result.multiples[2], 6.0)


def test_non_competing_recipes_independent():
    """Each recipe uses a different resource; both run at full supply."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    gas = Material(name="Gas", tags=GAS)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(10 * ore + 8 * gas, name="Supply", max_multiples=1)
    sell_ore = Recipe(-1 * ore + 1 * dollar, name="Sell Ore")
    sell_gas = Recipe(-1 * gas + 2 * dollar, name="Sell Gas")
    result = solve(set(), [supply, sell_ore, sell_gas], dollar)
    assert result.status == 0
    assert np.isclose(result.objective, 26.0)


def test_integer_only_recipe_forces_whole_multiples():
    """An integer_only recipe's multiple must be a whole number, even when
    that leaves supply unused."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-2 * ore + 3 * dollar, name="Sell", integer_only=True)
    result = solve(set(), [supply, sell], dollar)
    assert result.status == 0
    # 5 ore / 2 per multiple = 2.5, but integer_only rounds down to 2.
    assert np.isclose(result.multiples[1], 2.0)
    assert np.isclose(result.objective, 6.0)


def test_material_to_maximize_not_found_raises():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    other = Material(name="Unrelated")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    with pytest.raises(ValueError, match="not found"):
        solve(set(), [supply], other)


def test_material_balance_matches_recipe_matrix():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    result = solve(set(), [supply, sell], dollar)
    plus_amount, net_amount = material_balance(result, result.multiples)
    ore_idx = result.materials.index(ore)
    dollar_idx = result.materials.index(dollar)
    # All 5 ore produced then fully consumed -> net 0, gross produced 5.
    assert np.isclose(net_amount[ore_idx], 0.0, atol=1e-9)
    assert np.isclose(plus_amount[ore_idx], 5.0)
    # $ is only ever produced, never consumed -> net == gross == objective.
    assert np.isclose(net_amount[dollar_idx], result.objective)
    assert np.isclose(plus_amount[dollar_idx], result.objective)


def test_foo_bar_scenario_matches_reference_solution():
    """Reproduces factorylib.endfield.main's former test_main() demo scenario
    (a free-materials grant, a cheap linear conversion, a scarce integer-only
    "special offer", and an inefficient pure-Barium fallback conversion) and
    pins the known-optimal answer: fully use the integer special offer (bound
    to 1 whole run by Fooium supply), spend all remaining Fooium via the 1:1
    conversion, then mop up leftover Barium via the inefficient conversion.
    """
    fooium = Material(name="Fooium", unit="/min", tags=SOLID)
    barium = Material(name="Barium", unit="/min", tags=SOLID)
    foobarium = Material(name="Foobarium", unit="/min", tags=SOLID)
    free_materials = Recipe(
        fooium * 3 + barium * 7, name="Starting Materials", max_multiples=1
    )
    foo_plus_bar = Recipe(-fooium - barium + foobarium, name="Add Foo And Bar")
    special_offer = Recipe(
        -2 * fooium - barium + 4 * foobarium,
        name="Special Integer Reaction",
        max_multiples=2,
        integer_only=True,
    )
    pure_barium = Recipe(-barium + 0.1 * foobarium, name="Inefficient Barium Conversion")
    all_recipes = [free_materials, foo_plus_bar, special_offer, pure_barium]
    result = solve([fooium, barium, foobarium], all_recipes, foobarium)
    assert result.status == 0
    assert np.isclose(result.objective, 5.5)
    assert np.allclose(result.multiples, [1.0, 1.0, 1.0, 5.0])
