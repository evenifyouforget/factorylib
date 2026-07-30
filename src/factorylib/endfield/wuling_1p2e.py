"""Historical Wuling 1.2 -> 1.2d -> 1.2e recipe model.

Ported from the retired ``tests/wuling/*`` scenario tests
(``_helpers.py``'s hand-derived formula set, the most-refined link being
``test_wuling_1p2e.py``'s ``_make_1p2e_formulas``) onto the current
``Material``/``Recipe``/``factorylib.optimize.solve`` engine. This is NOT
ported from ``factorylib.endfield.main``, which only ever modeled 1.4 and
never modeled 1.2e at all. It's deliberately sized to reproduce exactly the
"headline" dollar figures those tests documented, not a full port of every
scenario they exercised: their many per-formula ad-hoc caps (like
``sc_cap_2``) or banned-formula variants aren't part of the core
resource-graph progression, and the Xiranite/Jade Gourd event extensions
are a separate, later model this doesn't attempt.

One shared resource graph (``build_1p2e_recipes``) reproduces all three
historical scenarios purely by varying its keyword arguments. The
1.2 -> 1.2d -> 1.2e lineage is genuinely the *same* resource graph,
extended incrementally, not three different models:

- ``test_1p2e_full`` (206735/146): full 1.2e, both Purification Building
  and Test Area Purification Node enabled, 1.2d supply/forge numbers.
- ``test_1p2e_equiv_1p2d`` (2823/2, == the historical 1.2d baseline): same
  supply, but with ``purify_node_max_multiples=0`` -- without the Test
  Area Purification Node, 1.2e collapses to 1.2d exactly.
- ``test_1p2e_equiv_1p2_full`` (2229/2, == the historical 1.2 "full"
  baseline): 1.2 full's own (smaller) supply/forge numbers, again with
  ``purify_node_max_multiples=0``.

This model has no power/Watt dimension at all, since every historical
formula is a pure resource conversion, unlike ``main.py``'s 1.4 model.

Forge of the Sky allocation: the original test scaffolding
(``_helpers.py``'s ``_search``) brute-forces "z forges produce Xiranite,
the rest cap Heavy Xiranite's rate" as an outer Python loop over
``max_forges + 1`` candidates. Here that split is a proper MILP choice
instead. A shared ``ForgeAllocation`` counted resource (supply =
``max_forges``) is consumed by both the "Forge of the Sky: Xiranite"
recipe and the "Heavy Xiranite" recipe itself, via the
counter/Assign-Allocation pattern ``main.py``'s own ``std_building`` uses
for its ``integer_inputs``.

Metatransfer is similarly converted from ``_helpers.py``'s
``METATRANSFERS`` outer-loop choice (either +50 Originium Ore or +25
Ferrium Ore) into a one-shot ``MetatransferAllocation`` MILP choice
between two recipes, mirroring ``main.py``'s own metatransfer-option loop.
"""

from __future__ import annotations

from math import inf

from factorylib.material import (
    LIQUID,
    SOLID,
    VIRTUAL,
    Material,
    MaterialExpression,
    Recipe,
)

_PER_MINUTE = "/min"


def build_1p2e_recipes(
    originium_ore: float,
    ferrium_ore: float,
    cuprium_ore: float,
    max_forges: int,
    *,
    purify_building_max_multiples: float = inf,
    purify_node_max_multiples: float = inf,
) -> tuple[set[Material], set[Recipe], Material]:
    all_recipes: list[Recipe] = []

    def std_solid(name: str) -> Material:
        return Material(name=name, unit=_PER_MINUTE, tags=SOLID)

    def std_liquid(name: str) -> Material:
        return Material(name=name, unit=_PER_MINUTE, tags=LIQUID)

    def std_alloc(name: str) -> Material:
        return Material(name=f"Allocation: {name}", tags=VIRTUAL)

    def counted_recipe(
        name: str, expr: MaterialExpression, integer_inputs: Material
    ) -> None:
        """A recipe gated by a shared, integer-counted resource -- the
        counter/Assign-Allocation split (see module docstring)."""
        counter = std_alloc(name)
        all_recipes.append(
            Recipe(
                counter - integer_inputs,
                name=f"Assign Allocation: {name}",
                integer_only=True,
            )
        )
        all_recipes.append(Recipe(-counter + expr, name=name))

    Dollar = Material(name="$", unit=_PER_MINUTE, tags=VIRTUAL)
    OriginiumOre = std_solid("Originium Ore")
    FerriumOre = std_solid("Ferrium Ore")
    CupriumOre = std_solid("Cuprium Ore")
    Cuprium = std_solid("Cuprium")
    Xiranite = std_solid("Xiranite")
    Sewage = std_liquid("Sewage")
    Effluent = std_liquid("Effluent")
    Inert = std_liquid("Inert")
    ForgeAllocation = std_alloc("Forge of the Sky")
    MetatransferAllocation = std_alloc("Metatransfer")

    all_recipes.append(
        Recipe(
            originium_ore * OriginiumOre
            + ferrium_ore * FerriumOre
            + cuprium_ore * CupriumOre
            + max_forges * ForgeAllocation
            + 1 * MetatransferAllocation,
            name="Starting Materials",
            max_multiples=1,
        )
    )
    all_recipes.append(
        Recipe(
            -MetatransferAllocation + 50 * OriginiumOre,
            name="Choose Metatransfer: Originium Ore",
            integer_only=True,
        )
    )
    all_recipes.append(
        Recipe(
            -MetatransferAllocation + 25 * FerriumOre,
            name="Choose Metatransfer: Ferrium Ore",
            integer_only=True,
        )
    )

    counted_recipe("Forge of the Sky: Xiranite", 30 * Xiranite, ForgeAllocation)
    counted_recipe(
        "Heavy Xiranite",
        -60 * Xiranite - 30 * Effluent + 27 * 6 * Dollar,
        ForgeAllocation,
    )

    all_recipes.append(
        Recipe(
            -30 * CupriumOre + 30 * Cuprium + 30 * Sewage,
            name="Convert Cuprium Ore",
        )
    )
    all_recipes.append(
        Recipe(
            -30 * Xiranite - 30 * Sewage + 30 * Effluent + 30 * Inert,
            name="React Xiranite with Sewage",
        )
    )
    all_recipes.append(
        Recipe(
            -240 * OriginiumOre
            - 30 * FerriumOre
            - 60 * Effluent
            + 30 * Sewage
            + 54 * 6 * Dollar,
            name="SC Wuling Battery",
        )
    )
    all_recipes.append(
        Recipe(
            -30 * Xiranite - 180 * OriginiumOre + 25 * 6 * Dollar,
            name="LC Wuling Battery",
        )
    )
    all_recipes.append(
        Recipe(
            -30 * FerriumOre - 240 * Cuprium + 30 * Sewage + 48 * 6 * Dollar,
            name="Hetonite Part",
        )
    )
    all_recipes.append(
        Recipe(-120 * Cuprium + 22 * 6 * Dollar, name="Yazhen Syringe A")
    )
    all_recipes.append(
        Recipe(-120 * FerriumOre + 16 * 6 * Dollar, name="Yazhen Syringe C")
    )
    all_recipes.append(Recipe(-Xiranite + Dollar, name="Sell Xiranite"))
    all_recipes.append(Recipe(-Cuprium + Dollar, name="Sell Cuprium"))
    all_recipes.append(
        Recipe(
            -120 * Inert + 30 * Effluent,
            name="Purification Building",
            max_multiples=purify_building_max_multiples,
        )
    )
    all_recipes.append(
        Recipe(
            -30 * Sewage + Effluent,
            name="Test Area Purification Node",
            max_multiples=purify_node_max_multiples,
        )
    )

    return set(), set(all_recipes), Dollar
