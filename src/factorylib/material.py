"""Symbolic Material/Recipe algebra.

This is the core modeling layer used to describe a production system: a
handful of ``Material`` instances (resources, intermediates, virtual
bookkeeping quantities like "$" or power) combined via ``+`` / ``-`` / ``*``
into linear expressions, and ``Recipe`` objects that pair a (materials in) -->
(materials out) expression with a name and optional run-count bounds.

The only legal shape of a Material expression is a linear combination
``c_1*m_1 + c_2*m_2 + ... + c_n*m_n``, where each ``c_i`` is a plain number
and each ``m_i`` is an elementary ``Material`` leaf -- there is no other way
to legally combine materials (multiplying two Material expressions together
is never legal, since neither side is a plain number). ``MaterialExpression``
is the abstract base for the two concrete shapes this takes: a bare
``Material`` (the trivial one-term combination ``1*self``) or a
``LinearCombinationMaterial`` (any other combination, stored flat as a
``{Material: coefficient}`` dict rather than a tree). Arithmetic always
normalizes to one of these two -- adding/multiplying always merges into (or
rebuilds) a flat dict, immediately combining matching terms and folding
constants, rather than building up a nested tree that would need a separate
flattening pass later. ``substitute()`` evaluates an expression against a
mapping from each leaf ``Material`` to a concrete value (typically a numpy
unit basis vector, when building the recipe matrix for
:mod:`factorylib.optimize`) -- generically, via ``terms()``, so neither
concrete subclass needs its own ``substitute()``/``gather_materials()``.

Because the same expression is evaluated over ``Material`` leaves, plain
numbers, and numpy arrays interchangeably, the value type flowing through
``substitute``/the operator overloads is genuinely dynamic. Rather than force
an imprecise or overly-broad Union everywhere, the few dynamic seams are
typed ``Any`` explicitly -- a deliberate, narrow strictness cut rather than
an oversight.

The flat representation makes printing straightforward and always correct:
``LinearCombinationMaterial.__str__`` always shows a term's coefficient and
unit (e.g. ``"3/min Ore"``, even at coefficient 1: ``"1/min Ore"``) and
always renders a negative coefficient as ``" - "`` rather than ``" + -"``,
with no special-casing or sign-detection needed -- every term's sign is
already known exactly, since matching terms were already merged when the
expression was built. An earlier version tried to bolt this kind of
prettification onto a tree of ``AddMaterial``/``MulMaterial`` nodes (plus a
separate ``simplify()`` pass to flatten nested trees so sign-detection could
see through them), which produced inconsistent output whenever the detection
didn't quite line up with how an expression was actually built -- e.g.
``"30 x Foo + -Bar + Baz x -40 x 2"``. Storing terms flat from the start
removes the class of bug entirely, rather than papering over it.
"""

from __future__ import annotations

from typing import Any

# Material "phase"/kind tags. These are plain strings that get concatenated
# (e.g. ``VIRTUAL + HIDDEN``) and membership-tested (e.g. ``HIDDEN in tags``)
# -- deliberately not an enum/flag type, to match how callers build them up.
SOLID = "S"
LIQUID = "L"
GAS = "G"
VIRTUAL = "V"
HIDDEN = "H"

_unique_counter = 0


class MaterialExpression:
    """Abstract base for anything usable in a linear Recipe expression: a
    linear combination of elementary ``Material`` leaves. The two concrete
    shapes are ``Material`` itself (a trivial one-term combination) and
    ``LinearCombinationMaterial`` (any other combination) -- see module
    docstring. Concrete subclasses only need to implement ``terms()`` and
    ``__str__``; arithmetic, ``substitute()``, and ``gather_materials()``
    are all defined here generically in terms of ``terms()``.
    """

    def terms(self) -> dict["Material", Any]:
        """This expression as a ``{elementary Material: coefficient}``
        dict. Never includes zero-coefficient entries."""
        raise NotImplementedError

    def gather_materials(self) -> set["Material"]:
        return set(self.terms())

    def substitute(self, subs_dict: dict["Material", Any]) -> Any:
        result: Any = 0
        for material, coeff in self.terms().items():
            result = result + coeff * subs_dict[material]
        return result

    def __add__(self, other: Any) -> MaterialExpression:
        if isinstance(other, MaterialExpression):
            merged = dict(self.terms())
            for material, coeff in other.terms().items():
                merged[material] = merged.get(material, 0) + coeff
            return _combination(merged)
        if not other:
            return self
        raise TypeError(
            f"cannot add {other!r} to a Material expression -- only 0 or "
            "another Material expression is legal"
        )

    def __radd__(self, other: Any) -> MaterialExpression:
        return self + other  # type: ignore[no-any-return]

    def __sub__(self, other: Any) -> MaterialExpression:
        if isinstance(other, MaterialExpression):
            return self + (other * -1)
        return self + other  # type: ignore[no-any-return]

    def __rsub__(self, other: Any) -> MaterialExpression:
        return (self * -1) + other  # type: ignore[no-any-return]

    def __mul__(self, other: Any) -> MaterialExpression:
        if not isinstance(other, (int, float)):
            raise TypeError(
                f"cannot multiply a Material expression by {other!r} -- only "
                "by a plain number (multiplying two Material expressions "
                "together is never legal)"
            )
        if other == 1:
            return self
        return _combination(
            {material: coeff * other for material, coeff in self.terms().items()}
        )

    def __rmul__(self, other: Any) -> MaterialExpression:
        return self * other  # type: ignore[no-any-return]

    def __neg__(self) -> MaterialExpression:
        return self * -1


def _combination(terms: dict["Material", Any]) -> MaterialExpression:
    """Build the canonical MaterialExpression for `terms`: a bare Material
    if it reduces to exactly one term at coefficient 1, otherwise a
    LinearCombinationMaterial (possibly with zero terms, representing the
    zero expression -- e.g. a 0 W building's power draw)."""
    nonzero = {material: coeff for material, coeff in terms.items() if coeff != 0}
    if len(nonzero) == 1:
        ((material, coeff),) = nonzero.items()
        if coeff == 1:
            return material
    return LinearCombinationMaterial(nonzero)


class Material(MaterialExpression):
    """A named quantity (a resource, intermediate, or virtual bookkeeping
    value) that can be combined into linear expressions via ``+``/``-``/``*``.
    """

    def __init__(
        self,
        _id: int | None = None,
        name: str | None = None,
        unit: str = "",
        tags: str = "",
    ) -> None:
        global _unique_counter
        if _id is None:
            _id = _unique_counter
            _unique_counter += 1
        if name is None:
            name = f"Anonymous Material #{_unique_counter}"
        self._id = _id
        self.name = name
        self.unit = unit
        self.tags = tags

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Material) and self._id == other._id

    def __hash__(self) -> int:
        return hash((self._id, 12345))  # magic number

    def __lt__(self, other: Material) -> bool:
        return self.name < other.name

    def __le__(self, other: Material) -> bool:
        return self.name <= other.name

    def terms(self) -> dict[Material, Any]:
        return {self: 1}

    def __str__(self) -> str:
        return f"{self.name}"


class LinearCombinationMaterial(MaterialExpression):
    """A linear combination of two or more elementary Materials (or a
    single Material at a coefficient other than 1, or zero materials at
    all -- see ``_combination()``), stored flat rather than as a tree."""

    def __init__(self, terms: dict[Material, Any]) -> None:
        self._terms = terms

    def terms(self) -> dict[Material, Any]:
        return self._terms

    def __str__(self) -> str:
        if not self._terms:
            return "0"
        parts: list[str] = []
        for material, coeff in self._terms.items():
            term = f"{abs(coeff)}{material.unit} {material.name}"
            if not parts:
                parts.append(f"-{term}" if coeff < 0 else term)
            else:
                parts.append(f" - {term}" if coeff < 0 else f" + {term}")
        return "".join(parts)


def substitute(expr: Any, subs_dict: dict[Material, Any]) -> Any:
    if isinstance(expr, (int, float)):
        return expr
    return expr.substitute(subs_dict)


class Recipe:
    """A named (materials in) --> (materials out) transformation.

    Args:
        expression: a Material expression where negative-coefficient terms
            are consumed inputs and positive-coefficient terms are produced
            outputs (e.g. ``-2 * Ore + 1 * Bar``).
        name: human-readable label.
        max_multiples: upper bound on how many times this recipe can run
            (default unbounded).
        integer_only: if True, this recipe's run count is constrained to
            whole multiples only.
    """

    def __init__(
        self,
        expression: MaterialExpression,
        name: str,
        max_multiples: float = float("inf"),
        integer_only: bool = False,
    ) -> None:
        self.expression = expression
        self.name = name
        self.max_multiples = max_multiples
        self.integer_only = integer_only

    def gather_materials(self) -> set[Material]:
        return gather_materials(self.expression)


def gather_materials(
    expr: MaterialExpression | Recipe | list[Any] | int | float,
) -> set[Material]:
    if isinstance(expr, list):
        result: set[Material] = set()
        for ex in expr:
            result |= gather_materials(ex)
        return result
    if isinstance(expr, (int, float)):
        return set()
    return expr.gather_materials()
