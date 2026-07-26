"""Symbolic Material/Recipe algebra.

This is the core modeling layer used to describe a production system: a
handful of ``Material`` instances (resources, intermediates, virtual
bookkeeping quantities like "$" or power) combined via ``+`` / ``-`` / ``*``
into linear expressions, and ``Recipe`` objects that pair a (materials in) -->
(materials out) expression with a name and optional run-count bounds.

``Material`` overloads the ordinary arithmetic operators to build a small
expression tree (``AddMaterial`` / ``MulMaterial`` nodes over ``Material``
leaves and numeric coefficients) rather than doing any numeric work itself.
``substitute()`` later evaluates such a tree against a mapping from each leaf
``Material`` to a concrete value (typically a numpy unit basis vector, when
building the recipe matrix for :mod:`factorylib.optimize`).

Because the same expression tree is evaluated over ``Material`` leaves,
plain numbers, and numpy arrays interchangeably, the value type flowing
through ``substitute``/the operator overloads is genuinely dynamic. Rather
than force an imprecise or overly-broad Union everywhere, the few dynamic
seams are typed ``Any`` explicitly (see module docstring notes below) --
a deliberate, narrow strictness cut rather than an oversight.
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


class Material:
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

    def substitute(self, subs_dict: dict[Material, Any]) -> Any:
        return subs_dict[self]

    def __add__(self, other: Any) -> Any:
        if not other:
            return self
        return AddMaterial(self, other).simplify()

    def __radd__(self, other: Any) -> Any:
        return self + other

    def __sub__(self, other: Any) -> Any:
        return self + other * -1

    def __rsub__(self, other: Any) -> Any:
        return other + self * -1

    def __mul__(self, other: Any) -> Any:
        if other == 1:
            return self
        return MulMaterial(self, other).simplify()

    def __rmul__(self, other: Any) -> Any:
        return self * other

    def __neg__(self) -> Any:
        return self * -1

    def gather_materials(self) -> set[Material]:
        return {self}

    def __str__(self) -> str:
        return f"{self.name}"

    def simplify(self) -> Material:
        return self

    def is_negative(self) -> bool:
        return False

    def split(self) -> tuple[Any, Any]:
        return (0, self)


class AddMaterial(Material):
    """A sum of two material expressions (built by ``Material.__add__``)."""

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def substitute(self, subs_dict: dict[Material, Any]) -> Any:
        return substitute(self.lhs, subs_dict) + substitute(self.rhs, subs_dict)

    def gather_materials(self) -> set[Material]:
        return gather_materials(self.lhs) | gather_materials(self.rhs)

    def __str__(self) -> str:
        if self.rhs.is_negative():
            return f"{self.lhs} - {-1 * self.rhs}"
        return f"{self.lhs} + {self.rhs}"

    def simplify(self) -> Material:
        # try to expand sum
        queue: list[Any] = [self.lhs, self.rhs]
        others: list[Any] = []
        while queue:
            x = queue.pop()
            if isinstance(x, AddMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        result = others[0]
        for x in others[1:]:
            result = AddMaterial(result, x)
        return result  # type: ignore[no-any-return]

    def split(self) -> tuple[Any, Any]:
        queue: list[Any] = [self.lhs, self.rhs]
        others: list[Any] = []
        while queue:
            x = queue.pop()
            if isinstance(x, AddMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        pos = []
        neg = []
        for x in others:
            if x.is_negative():
                neg.append(-x)
            else:
                pos.append(x)
        return sum(neg), sum(pos)


class MulMaterial(Material):
    """A product of two material expressions (built by ``Material.__mul__``).

    In practice one side is always a plain numeric coefficient and the other
    a ``Material``/expression (e.g. ``3 * OriginiumOre``), but the algebra
    itself doesn't assume that.
    """

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def substitute(self, subs_dict: dict[Material, Any]) -> Any:
        return substitute(self.lhs, subs_dict) * substitute(self.rhs, subs_dict)

    def gather_materials(self) -> set[Material]:
        return gather_materials(self.lhs) | gather_materials(self.rhs)

    def __str__(self) -> str:
        lhs = self.lhs
        rhs = self.rhs
        if isinstance(rhs, (int, float)) and isinstance(lhs, Material):
            lhs, rhs = rhs, lhs
        if isinstance(lhs, (int, float)) and type(rhs) is Material:
            return f"{lhs}{rhs.unit} {rhs.name}"
        if lhs == -1:
            return f"-{rhs}"
        return f"{lhs}×{rhs}"

    def simplify(self) -> Material:
        # try to expand product
        queue: list[Any] = [self.lhs, self.rhs]
        constant: Any = 1
        others: list[Any] = []
        while queue:
            x = queue.pop()
            if isinstance(x, (int, float)):
                constant *= x
            elif isinstance(x, MulMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        result = others[0]
        for x in others[1:]:
            result = MulMaterial(result, x)
        result = MulMaterial(constant, result)
        return result

    def is_negative(self) -> bool:
        lhs = self.lhs
        rhs = self.rhs
        if isinstance(rhs, (int, float)) and isinstance(lhs, Material):
            lhs, rhs = rhs, lhs
        if isinstance(lhs, (int, float)) and type(rhs) is Material:
            return bool(lhs < 0)
        return False


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
        expression: Any,
        name: str,
        max_multiples: float = float("inf"),
        integer_only: bool = False,
    ) -> None:
        self.expression = expression.simplify()
        self.name = name
        self.max_multiples = max_multiples
        self.integer_only = integer_only

    def gather_materials(self) -> set[Material]:
        return gather_materials(self.expression)

    def nice_expression_str(self) -> str:
        neg, pos = self.expression.split()
        return f"{neg} --> {pos}"


def gather_materials(expr: Any) -> set[Material]:
    if isinstance(expr, list):
        result: set[Material] = set()
        for ex in expr:
            result |= gather_materials(ex)
        return result
    if isinstance(expr, (int, float)):
        return set()
    return expr.gather_materials()  # type: ignore[no-any-return]
