# Understanding the mechanics

The depot acts as a global shared inventory.

Raw materials (namely ores) are harvested in the wild, and directly enter the depot at a fixed rate. Ores (or potentially other materials) are then drawn from the depot by a depot unloader. Materials can be stashed back in the depot by a depot loader or (more commonly for space efficiency) a protocol stash.

Belts (solids) flow at 30 items/min, while pipes (liquids) flow at 120 items/min.

This means for example, if the total originium ore income was 120/min, we could supply 4 full belts, but if we tried to draw 5 belts, each belt would only run at 4/5 capacity.

Since depot unloaders take up a lot of precious space, it is generally preferable to pipe outputs of some buildings directly into other buildings. However, if that wiring would take up too much space, it is preferable to instead stash in the depot and place a depot unloader elsewhere, effectively teleporting the items.

Stashing and unloading also allows the depot to act like a giant many-way balancer. However, the turn-taking behaviour of depot balancing is poorly understood, so this is generally only used in simple situations, such as drawing the exact income of a resource, or overdrawing by a fraction of a belt when all the outputs are going to the same production formula anyway.

Accumulating items in the depot intentionally may be used so those items can be sold, used for delivery jobs, or used for crafting.

# Physical construction of factories

In the simplest case, the exact amount of inputs needed is drawn from depot unloaders, and those inputs only ever reach one output product type, with no intermediate materials ever entering the depot. In this case, the production of each final material is independent, and there is no interaction.

However, Wuling's case is significantly more interconnected, with there being notable loops for sewage and acid, and multiple end products wanting to use effluent (which benefits from a central processing area).

When presented with a factory plan with difficult fractions, we have a few tools.

## Exact splitting

Suppose we wanted to make 10/min Ferrium Bottle. The raw input needed for this is 20/min Ferrium Ore. So we can draw 1 full belt = 30/min, and take 2/3 of it.

This also works with intermediate products.

## Rate limiting

Suppose we wanted to make 10/min Ferrium Bottle. The raw input needed for this is 20/min Ferrium Ore.

We can set up a different splitter/converger network, this time on the output (just to show we can do something that exact splitting can't), which limits the flow to 10/min.

Backpressure will result in input consumption being limited to 20/min.

## Priority overflow

Suppose we then also wanted to make 10/min Ferrium Part.

We could set up a perfect priority splitter, with the previous Ferrium Bottle sub-factory having precedence. So it will take 20/min, and the remaining 10/min goes to making Ferrium Part.

This can also be done using the depot. For example, depot unloader -> refining -> protocol stash -> [bottle sub-factory]. If the bottle sub-factory consumes 30/min Ferrium, items pass through the protocol stash without ever getting stashed. If the bottle sub-factory consumes, less, say, it only consumes 20/min, then the protocol stash accumulates items at a rate of 10/min. Every 10 seconds, the protocol stash attempts to send its inventory contents to the depot, which is this 10/min. If the downstream factory was more clogged and consumed 0/min, the protocl stash now stashes 30/min.

We can then draw one belt of Ferrium for the parts sub-factory. Since we demand 30/min, we can always consume however much Ferrium is generated.

In this example, the parts sub-factory can "scale up" production dynamically in response to another area of the factory getting clogged. But this may not always be possible - for example, if we instead set up 3 of those bottle factories for some reason, we could take just 1 belt of Ferrium, assuming that 30/min demand would be enough to consume the supply, but this cannot scale up at all.

If the parts factory gets clogged, Ferrium may now accumulate in the depot. This might be okay since that's Ferrium that wouldn't be used anyway, but it's slightly not ideal.

## Limiting reactions

Just limit one ingredient.

## Virtual limits

This technique is entirely non-obvious, because it happens in the planning stage.

Suppose we have sub-factories A and B. A needs 21.XXX/min of Ferrium Ore, and B needs 6.XXX/min of Ferrium Ore. These might be really nasty fractions. Suppose also we have 30/min Ferrium Ore income.

Well, what if we just supplied A with 22.5/min (= 3/4 of a belt), and B with 7.5/min (= 1/4 of a belt)? This might result in more products being produced than would be produced with exact fractions, but... we probably don't care. If those are extra batteries that end up being consumed by delivery jobs, it's fine, our virtual plans didn't need those batteries at all anyway, we're fine without them.

So we may try to round up fractions, if we have input slack. The appeal of this is, of course, to get nice fractions.

We do need to be careful, as this can have unintended consequences, such as:

- clogging a sewage loop that would have had excess capacity otherwise
- using too much of some other material that would have been limited by a limiting reagent, that has priority overflow to some other component that isn't expecting the priority branch to take so much

So perhaps this technique will be usable fully by proving that it should not break anything, or it is usable only if we generate a physical layout, or we can generate it with warnings, or something else?
