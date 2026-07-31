"""Tests for factorylib.material (Material/LinearCombinationMaterial/Recipe
algebra), ported from factorylib.endfield.main -- the design oracle."""

from __future__ import annotations

import numpy as np
import pytest

from factorylib.material import (
    GAS,
    HIDDEN,
    LIQUID,
    SOLID,
    VIRTUAL,
    LinearCombinationMaterial,
    Material,
    Recipe,
    gather_materials,
    substitute,
)


def test_material_identity_by_id_not_name():
    a = Material(name="Ore")
    b = Material(name="Ore")
    assert a != b  # distinct auto-assigned ids
    assert a == a


def test_material_explicit_id_equality():
    a = Material(_id=42, name="Ore")
    b = Material(_id=42, name="Different Name")
    assert a == b
    assert hash(a) == hash(b)


def test_material_default_name_is_anonymous():
    m = Material()
    assert m.name.startswith("Anonymous Material #")


def test_material_str_uses_name():
    m = Material(name="Cuprium Ore")
    assert str(m) == "Cuprium Ore"


def test_material_ordering_by_name():
    a = Material(name="Alpha")
    b = Material(name="Beta")
    assert a < b
    assert a <= b
    assert not b < a


def test_material_tags_default_empty_and_combinable():
    m = Material(name="X", tags=VIRTUAL + HIDDEN)
    assert HIDDEN in m.tags
    assert VIRTUAL in m.tags
    default = Material(name="Y")
    assert default.tags == ""


def test_material_terms_is_trivial_one_term_combination():
    ore = Material(name="Ore")
    assert ore.terms() == {ore: 1}


def test_add_material_builds_sum_and_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = ore + bar
    assert isinstance(expr, LinearCombinationMaterial)
    assert expr.terms() == {ore: 1, bar: 1}
    # Every term shows its unit/coefficient elegantly, even at 1 -- there's
    # no special-casing needed since the representation is already flat.
    assert str(expr) == "1/min Ore + 1/min Bar"


def test_add_material_with_zero_returns_self():
    ore = Material(name="Ore")
    assert (ore + 0) is ore
    assert (0 + ore) is ore


def test_add_nonzero_plain_number_raises():
    """The only legal expression shape is a linear combination of
    Materials -- a bare nonzero constant term has no physical meaning
    here (unlike 0, which is just the additive identity, needed for
    sum() to work)."""
    ore = Material(name="Ore")
    with pytest.raises(TypeError, match="cannot add"):
        ore + 5
    with pytest.raises(TypeError, match="cannot add"):
        5 + ore


def test_sub_renders_with_correct_sign_and_unit():
    """Subtraction is addition of a -1 coefficient, and the flat
    representation always renders a negative term with a real "-" and
    its unit -- no sign-detection bugs possible, since the coefficient's
    sign is already known exactly."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = ore - bar
    assert expr.terms() == {ore: 1, bar: -1}
    assert str(expr) == "1/min Ore - 1/min Bar"


def test_rsub_with_nonzero_plain_number_on_left_raises():
    """Same "only 0 or another Material expression" invariant applies via
    __rsub__ as via __add__/__sub__."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    with pytest.raises(TypeError, match="cannot add"):
        10 - ore


def test_sub_zero_returns_self():
    ore = Material(name="Ore")
    assert (ore - 0) is ore


def test_mul_material_coefficient_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = 3 * ore
    assert isinstance(expr, LinearCombinationMaterial)
    assert str(expr) == "3/min Ore"


def test_mul_by_one_returns_self():
    ore = Material(name="Ore")
    assert (ore * 1) is ore
    assert (1 * ore) is ore


def test_mul_by_zero_is_the_zero_expression():
    """0 W times a material (e.g. a 0 W building's power draw) must still
    produce a valid, further-combinable expression, not a crash."""
    watt = Material(name="W", tags=VIRTUAL)
    expr = 0 * watt
    assert expr.terms() == {}
    assert str(expr) == "0"
    assert (expr + watt).terms() == {watt: 1}


def test_mul_two_material_expressions_raises():
    """Multiplying two Material expressions together is never legal --
    only a plain number times a Material expression is."""
    ore = Material(name="Ore")
    bar = Material(name="Bar")
    with pytest.raises(TypeError, match="cannot multiply"):
        ore * bar


def test_neg_material_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = -ore
    assert str(expr) == "-1/min Ore"


def test_neg_of_compound_expression_negates_every_term():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = -(ore + bar)
    assert expr.terms() == {ore: -1, bar: -1}
    assert str(expr) == "-1/min Ore - 1/min Bar"


def test_gather_materials_over_nested_sums():
    a = Material(name="A")
    b = Material(name="B")
    c = Material(name="C")
    expr = (a + b) + c
    materials = gather_materials(expr)
    assert materials == {a, b, c}


def test_gather_materials_over_recipe_expression():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    metal = Material(name="Metal", unit="/min", tags=SOLID)
    recipe = Recipe(-2 * ore + 1 * metal, name="Refine")
    assert recipe.gather_materials() == {ore, metal}


def test_gather_materials_over_list_of_recipes():
    ore = Material(name="Ore")
    metal = Material(name="Metal")
    gas = Material(name="Gas", tags=GAS)
    r1 = Recipe(-1 * ore + 1 * metal, name="Refine")
    r2 = Recipe(1 * gas, name="Vent", max_multiples=1)
    assert gather_materials([r1, r2]) == {ore, metal, gas}


def test_recipe_defaults():
    ore = Material(name="Ore")
    recipe = Recipe(1 * ore, name="Free Ore")
    assert recipe.max_multiples == float("inf")
    assert recipe.integer_only is False


def test_substitute_evaluates_expression_against_basis_vectors():
    ore = Material(name="Ore")
    metal = Material(name="Metal")
    subs = {ore: np.array([1.0, 0.0]), metal: np.array([0.0, 1.0])}
    expr = -2 * ore + 3 * metal
    result = substitute(expr, subs)
    assert np.allclose(result, [-2.0, 3.0])


def test_substitute_passes_through_plain_numbers():
    assert substitute(5, {}) == 5
    assert substitute(2.5, {}) == 2.5


def test_gather_materials_passes_through_plain_numbers():
    assert gather_materials(5) == set()


def test_material_expression_terms_is_abstract():
    from factorylib.material import MaterialExpression

    with pytest.raises(NotImplementedError):
        MaterialExpression().terms()


def test_liquid_tag_constant_distinct_from_solid_and_gas():
    assert LIQUID != SOLID
    assert LIQUID != GAS
    assert len({SOLID, LIQUID, GAS, VIRTUAL, HIDDEN}) == 5


def test_material_lt_requires_material_operand():
    a = Material(name="A")
    with pytest.raises(AttributeError):
        _ = a < 5  # not a Material; .name lookup on int fails


def test_add_material_over_three_terms_merges_flat():
    """A 3+ term sum merges into one flat {Material: coefficient} dict,
    not a nested tree -- str() reads left-to-right in insertion order."""
    a = Material(name="A")
    b = Material(name="B")
    c = Material(name="C")
    expr = a + b + c
    assert isinstance(expr, LinearCombinationMaterial)
    assert expr.terms() == {a: 1, b: 1, c: 1}
    assert str(expr) == "1 A + 1 B + 1 C"


def test_repeated_addition_of_the_same_material_merges_coefficients():
    """Adding the same Material twice merges into a single term with a combined
    coefficient, rather than keeping two separate entries. This is what makes
    constant-folding automatic, with no separate simplify() pass."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = ore + ore
    assert expr.terms() == {ore: 2}
    assert str(expr) == "2/min Ore"


def test_nested_multiplication_constant_folds_automatically():
    """3 * (2 * ore) folds straight to a single 6-coefficient term. Unlike a
    tree representation, there's nothing to separately flatten: multiplying a
    LinearCombinationMaterial by a scalar just rescales its existing
    (already-merged) terms dict."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = 3 * (2 * ore)
    assert expr.terms() == {ore: 6}
    assert str(expr) == "6/min Ore"


def test_combination_of_combinations_merges_shared_materials():
    """(ore + bar) + (bar + metal) should merge the two `bar` terms into
    one coefficient-2 entry, not keep them separate."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    metal = Material(name="Metal", unit="/min", tags=SOLID)
    expr = (ore + bar) + (bar + metal)
    assert expr.terms() == {ore: 1, bar: 2, metal: 1}


def test_terms_that_cancel_out_are_dropped():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = (ore + bar) - bar
    # Cancels back down to the trivial one-term combination, i.e. `ore`
    # itself -- not a LinearCombinationMaterial with a zero bar term.
    assert expr is ore
