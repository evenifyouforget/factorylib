"""Tests for factorylib.material (Material/AddMaterial/MulMaterial/Recipe
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
    AddMaterial,
    Material,
    MulMaterial,
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


def test_add_material_builds_sum_and_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = ore + bar
    assert isinstance(expr, AddMaterial)
    assert str(expr) == "Ore + Bar"


def test_rsub_with_plain_number_on_left():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = 10 - ore
    assert str(expr) == "-1/min Ore + 10"


def test_add_material_with_zero_returns_self():
    ore = Material(name="Ore")
    assert (ore + 0) is ore
    assert (0 + ore) is ore


def test_sub_renders_as_plus_negative_coefficient():
    """__str__ never special-cases negation into a "-" -- subtraction is
    just addition of a -1 coefficient, printed plainly."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = ore - bar
    assert str(expr) == "Ore + -1/min Bar"


def test_mul_material_coefficient_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = 3 * ore
    assert isinstance(expr, MulMaterial)
    assert str(expr) == "3/min Ore"


def test_mul_by_one_returns_self():
    ore = Material(name="Ore")
    assert (ore * 1) is ore
    assert (1 * ore) is ore


def test_neg_material_str():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = -ore
    # Same "<coefficient><unit> <name>" branch as above applies here.
    assert str(expr) == "-1/min Ore"


def test_neg_of_compound_expression_has_no_special_case():
    """Negating a compound (AddMaterial) expression falls through to the
    plain "lhs×rhs" branch, since the "number, bare Material" special
    case only matches an exact Material, not an AddMaterial -- consistent
    (if not maximally pretty) rather than another special case."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    bar = Material(name="Bar", unit="/min", tags=SOLID)
    expr = -(ore + bar)
    assert str(expr) == "-1×Ore + Bar"


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


def test_liquid_tag_constant_distinct_from_solid_and_gas():
    assert LIQUID != SOLID
    assert LIQUID != GAS
    assert len({SOLID, LIQUID, GAS, VIRTUAL, HIDDEN}) == 5


def test_material_lt_requires_material_operand():
    a = Material(name="A")
    with pytest.raises(AttributeError):
        _ = a < 5  # not a Material; .name lookup on int fails


def test_add_material_over_three_terms_builds_a_nested_chain():
    """A 3+ term sum builds nested AddMaterial nodes (left-leaning, no
    flattening) -- str() still reads left-to-right correctly regardless,
    since nesting direction doesn't affect the "+"-joined text."""
    a = Material(name="A")
    b = Material(name="B")
    c = Material(name="C")
    expr = a + b + c
    assert gather_materials(expr) == {a, b, c}
    assert str(expr) == "A + B + C"


def test_mul_material_over_three_factors_is_not_constant_folded():
    """2 * 3 * ore is plain int arithmetic (2*3=6) before it ever reaches
    Material.__rmul__, so this still prints as "6/min Ore" -- but genuine
    nested MulMaterial (a Material expression times a Material
    expression, not two plain numbers) is no longer constant-folded."""
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = 2 * 3 * ore
    assert isinstance(expr, MulMaterial)
    assert str(expr) == "6/min Ore"

    nested = 3 * (2 * ore)
    assert str(nested) == "3×2/min Ore"


def test_mul_material_str_swaps_number_on_right():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    expr = MulMaterial(ore, 4)
    assert str(expr) == "4/min Ore"
