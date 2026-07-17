Suggest to replace the current specialized Component rewards with a Tier 1~4 Crafting Point.

Rather than need O(N^2) combinations, you can just have higher tier Crafting Point be able to convert to the next lower tier Crafting Point.

Based on the flexible crafting formulas, this should be accurate:

50 Pyrrolite Component -> 1 T4 Crafting Point
50 Hetonite Component -> 1 T3 Crafting Point
50 Cuprium Component -> 1 T2 Crafting Point
50 Xiranite Component -> 1 T1 Crafting Point
1 T4 Crafting Point -> 2 T3 Crafting Point
1 T3 Crafting Point -> 5 T2 Crafting Point
1 T2 Crafting Point -> 1 T1 Crafting Point

This should make it possible to achieve all 4 nonzero bonuses by only producing Pyrrolite Component.

99% discount is not modeled here. The hypothesis is that Pyrrolite Component will be preferred by the LP solver anyway. But if necessary, we can try to give it more incentive to produce Pyrrolite Component.
