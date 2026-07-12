Be brief. Ask if uncertain. Read code to understand conventions. Explore alternative designs.
git: commit allowed. commit often. Work on feature branch. Prefer surgical edits to minimize PR diff.

# Part 1 - Find breakpoint

Past PRs have historically done this manually (or rather, got the LLM agent to do it manually) which is wasteful.

I believe the methodology for finding a breakpoint is quite clear - it is a discontinuity in the function. An example is when hetonite is better to produce than yazhen A+C, and yazhen A+C production turns off.

From there, we try to find a simple fraction for the numerical result, if possible. Otherwise we return the raw numerical result.

This can all be bundled in a utility function, that finds discontinuities for a single-input function, for inputs within some bounds. Breakpoint analysis then becomes much easier.

# Part 2 - Find alternatives

We can wrap our original LP solver with a function that tries adjusting weights +/- some epsilon, and if it results in a new solution, return that solution as well, but with the outputs recalculated with unmodified weights (so no epsilon pollution in the result). It should return up to N solutions (configurable). This would help with cases where the reference values happen to sit on a breakpoint (discontinuity).

# Part 3 - Endfield module, standard formulas

Considering Endfield is our main target anyway, I believe it is reasonable to add code specialized to Endfield. But we should try to keep such highly specialized code separate from the core. Therefore I suggest nesting it under factorylib.endfield . factorylib.endfield can expose a main function such that if invoked as a CLI, it calculates an optimal solution and prints it. Configuration parameters should be exposed by CLI. Tied solutions (due to a discontinuity, see part 2) should be printed as well.

The first order of business is to make some class or utility function to set up the standard Wuling environment. This should allow configuration of which formulas are allowed, etc. with the default being 1.2e full (everything on).

Tests should make sure it is possible to replicate 1.2e full and some previous results, but by using this utility instead of redefining everything every time.

Future code will then be able to build on this.

A possibly non-exhaustive list of recipes:

Planting Unit: (null) -> 30 Sandleaf OR 30 Buckflower OR 30 Citrome OR 30 Aketine
Planting Unit: 30 Water -> 60 Yazhen OR 60 Jincao
Shredding Unit: 30 Yazhen -> 60 Yazhen Powder
Shredding Unit: 30 Jincao -> 60 Jincao Powder
Shredding Unit: 30 Cuprium -> 30 Cuprium Powder
Shredding Unit: 30 Ferrium -> 30 Ferrium Powder
Shredding Unit: 30 Amethyst -> 30 Amethyst Powder
Shredding Unit: 30 Originium Ore -> 30 Originium Powder
Shredding Unit: 30 Carbon -> 60 Carbon Powder
Shredding Unit: 30 Origocrust -> 30 Origocrust Powder
Shredding Unit: 30 Buckflower -> 60 Buckflower Powder
Shredding Unit: 30 Citrome -> 60 Citrome Powder
Shredding Unit: 30 Sandleaf -> 90 Sandleaf Powder
Shredding Unit: 30 Aketine -> 60 Aketine Powder
Fluid Pump: (null) -> 60 Water OR 60 Acid
Refining Unit: 30 Ferrium Ore -> 30 Ferrium
Refining Unit: 30 Amethyst Ore -> 30 Amethyst
Refining Unit: 30 Originium Ore -> 30 Origocrust
Refining Unit: 30 Dense Origocrust Powder -> 30 Packed Origocrust
Refining Unit: 30 Dense Originium Powder -> 30 Dense Origocrust Powder
Refining Unit: 30 Dense Ferrium Powder -> 30 Steel
Refining Unit: 30 Cryston Powder -> 30 Cryston Fiber
Refining Unit: 30 Dense Carbon Powder -> 30 Stabilized Carbon
Refining Unit: 30 Buckflower OR 30 Sandleaf -> 30 Carbon
Refining Unit: 30 Jincao OR 30 Yazhen -> 60 Carbon
Refining Unit: 30 Cuprium Ore + 30 Water -> 30 Cuprium + 30 Sewage
Reactor Crucible: 30 Yazhen Powder + 30 Water -> 30 Yazhen Solution
Reactor Crucible: 30 Jincao Powder + 30 Water -> 30 Jincao Solution
Reactor Crucible: 30 Xiranite + 30 Water -> 30 Liquid Xiranite
Reactor Crucible: 30 Heavy Xiranite + 30 Acid -> 30 Liquid Heavy Xiranite
Reactor Crucible: 30 Cuprium Powder + 30 Acid -> 30 Cuprium Solution
Reactor Crucible: 30 Liquid Xiranite + 30 Sewage -> 30 Xircon Effluent + 30 Inert Xircon Effluent
Reactor Crucible: 60 Xircon Effluent + 30 Ferrium Powder -> 30 Xircon + 30 Sewage
Reactor Crucible: 60 Hetonite Solution + 30 Ferrium Powder -> 30 Hetonite + 30 Sewage
Water Treatment Unit: 30 Sewage -> (null)
Moulding Unit: 60 Cuprium -> 30 Cuprium Bottle
Moulding Unit: 60 Ferrium -> 30 Ferrium Bottle
Moulding Unit: 60 Amethyst -> 30 Amethyst Bottle
Moulding Unit: 60 Steel -> 30 Steel Bottle
Moulding Unit: 60 Cryston -> 30 Cryston Bottle
Moulding Unit: 60 Hetonite -> 30 Hetonite Bottle
Fitting Unit: 30 Cuprium -> 30 Cuprium Part
Fitting Unit: 30 Ferrium -> 30 Ferrium Part
Fitting Unit: 30 Amethyst -> 30 Amethyst Part
Fitting Unit: 30 Steel -> 30 Steel Part
Fitting Unit: 30 Cryston -> 30 Cryston Part
Fitting Unit: 30 Hetonite -> 6 Hetonite Part
Filling Unit: 30 (some kind of Bottle) + 30 (some fluid) -> 30 (combination of that Bottle and that fluid)
Packaging Unit: 60 Cuprium Part + 30 Cuprium Bottle filled with Yazhen Solution -> 6 Yazhen Syringe A
Packaging Unit: 60 Cuprium Part + 30 Cuprium Bottle filled with Jincao Solution -> 6 Jincao Tea
Packaging Unit: 30 Amethyst Part + 6 Aketine Powder -> 6 Industrial Explosive
Packaging Unit: 30 Amethyst Part + 60 Originium Powder -> 6 LC Valley Battery
Packaging Unit: 60 Ferrium Part + 90 Originium Powder -> 6 SC Valley Battery
Packaging Unit: 60 Steel Part + 90 Dense Originium Powder -> 6 HC Valley Battery
Packaging Unit: 60 Ferrium Part + 30 Ferrium Bottle filled with Yazhen Solution -> 6 Yazhen Syringe C
Packaging Unit: 60 Ferrium Part + 30 Ferrium Bottle filled with Jincao Solution -> 6 Jincao Drink
Packaging Unit: 30 Xiranite + 90 Dense Originium Powder -> 6 LC Wuling Battery
Packaging Unit: 30 Xircon + 120 Dense Originium Powder -> 6 SC Wuling Battery
Grinding Unit: 60 Originium Powder + 30 Sandleaf Powder -> 30 Dense Originium Powder
Grinding Unit: 60 Origocrust Powder + 30 Sandleaf Powder -> 30 Dense Origocrust Powder
Grinding Unit: 60 Ferrium Powder + 30 Sandleaf Powder -> 30 Dense Ferrium Powder
Grinding Unit: 60 Amethyst Powder + 30 Sandleaf Powder -> 30 Cryston Powder
Grinding Unit: 60 Carbon Powder + 30 Sandleaf Powder -> 30 Dense Carbon Powder
Grinding Unit: 60 Buckflower Powder + 30 Sandleaf Powder -> 30 Ground Buckflower Powder
Grinding Unit: 60 Citrome Powder + 30 Sandleaf Powder -> 30 Ground Citrome Powder
Purification Unit: 120 Inert Xircon Effluent -> 30 Xircon Effluent + 30 Water
Purification Unit: 120 Cuprium Solution -> 30 Hetonite Solution + 30 Acid
Forge of the Sky: 60 Xiranite + 30 Xircon Effluent -> 6 Heavy Xiranite
Forge of the Sky: 60 Stabilized Carbon + 30 Water -> 30 Xiranite
Thermal Bank: 7.5 Originium Ore -> 50 W
Thermal Bank: 1.5 LC Valley Battery -> 220 W
Thermal Bank: 1.5 SC Valley Battery -> 420 W
Thermal Bank: 1.5 HC Valley Battery -> 1100 W
Thermal Bank: 1.5 LC Wuling Battery -> 1600 W
Thermal Bank: 1.5 SC Wuling Battery -> 3200 W
Gearing Unit: 60 Origocrust + 60 Ferrium -> 6 Ferrium Component
Gearing Unit: 60 Packed Origocrust + 60 Cryston Fiber -> 6 Cryston Component
Gearing Unit: 30 Origocrust + 30 Amethyst -> 6 Amethyst Component
Gearing Unit: 60 Packed Origocrust + 60 Xiranite -> 6 Xiranite Component
Gearing Unit: 60 Cuprium Part + 60 Xiranite -> 6 Cuprium Component
Gearing Unit: 12 Hetonite Part + 12 Heavy Xiranite -> 6 Hetonite Part
Test Area Purification Node: 30 Sewage -> 1 Xircon Effluent (max 12 multiples)
Sell: 1 Xiranite -> 1 Wuling Stock Bill
Sell: 1 Cuprium Part -> 1 Wuling Stock Bill
Sell: 1 Yazhen Syringe C -> 16 Wuling Stock Bill
Sell: 1 Jincao Drink -> 16 Wuling Stock Bill
Sell: 1 Yazhen Syringe A -> 22 Wuling Stock Bill
Sell: 1 Jincao Tea -> 22 Wuling Stock Bill
Sell: 1 LC Wuling Battery -> 25 Wuling Stock Bill
Sell: 1 Heavy Xiranite -> 27 Wuling Stock Bill
Sell: 1 Hetonite Part -> 48 Wuling Stock Bill
Sell: 1 SC Wuling Battery -> 54 Wuling Stock Bill
Craft Gear: 8000 Wuling Stock Bill + 50 Xiranite Component -> 1 Xiranite Component Gear
Craft Gear: 16000 Wuling Stock Bill + 50 Cuprium Component -> 1 Cuprium Component Gear
Craft Gear: 25000 Wuling Stock Bill + 50 Hetonite Component -> 1 Hetonite Component Gear

Final output must have net 0 or negative sewage production.

Please point out any recipes I may have apparently missed, or if any recipes don't line up (ex. due to typos).

# Part 4 - Expanded Wuling goals

Outposts "generate savings". We then sell goods, and these Wuling Stock Bills come from the savings. So that limits the Wuling Stock Bill income from selling, and with it, the amount of goods we can sell. We always want to generate more goods than we can sell.

- Sky King Flats Construction Site: 42000/hr
- Cardiac Remediation Station: 23400/hr

Note the different units: everything else is /min. So in practice this is 700/min and 390/min, and we saturate at 1090 Wuling Stock Bill (/min). This is likely to increase in a future update. So it should be configurable, and likewise for other parameters that may increase in the future.

It is generally preferable to have a significant excess. If, for example, we generate 1.1x the savings rate in sellable goods, the inventory is currently empty, and the current savings is at 90% of capacity, we can wait until savings are at 100%, then sell 11% of capacity, so savings is at 89% of capacity. We wait for 100% again, sell 12.1%... and it takes many cycles to get the savings down to 0. The problem is even worse at 1.01x, or 1.001x.

We also need to generate power. Currently power demands sit around 7000 W for an average player. This may also increase in the future. We want to produce this much or more worth of batteries.

Besides this, there are two other sinks for materials: gear/building crafting, and delivery jobs.

For the delivery job minigame (daily), we "pack goods". The box has a fixed capacity of let's say 14k, which we need to fill with any material. The quick top up button will select whatever good has the highest amount in the depot. We do this for 2 boxes. The basic way to fulfill this is by generating 28k daily of some very cheap material, such as Sandleaf Powder. Better if we generate 2 different materials, so the top 2 materials are things we don't care about. The depot has a limit, something like 80k (which may also increase in the future), so after depleting 14k of that material, we want it to be back up to 80k in less than 24 hours. In the future the box size will probably increase slightly, and we will have 3 boxes, so for ease of use we want 3 materials that we don't use for other goals that we can produce more than 14k of in 24 hours. Some safety margin is good in case we log in tomorrow ex. after 22 hours or 20 hours.

For gear crafting, it is not on a fixed schedule like daily tasks, but instead whenever the player wants to craft equipment. Historically this can be fulfilled by a very small amount of income: even 0.5/min of Cuprium Component means you can log off, log back in tomorrow, and have more than enough supplies to craft all the equipment you need. So we don't need a very big number, we just want it to not be 0. Likewise for other materials, there might be an occasional quest that needs it, so it can be helpful to have a slight excess, but other materials are generally in even less demand. The general order of demand is Parts > Components > Bottles > Xiranite/Heavy Xiranite > plants (Yazhen, Jincao, Sandleaf, Buckflower, Citrome, Aketine) > anything else.

All of this so far, and what comes next below, can be modeled with a nonlinear (or perhaps piecewise linear) fitness function of an overall production plan. Designing this function is a design task. You should explore multiple possibilities and weigh the merits of each, before settling on one final decision with rationale given.

Next consideration is simplicity. If, for example, the production plan asks for 19/96 of a Yazhen Syringe C production line, it may be difficult to produce this exact fraction, or actually impossible. The player may make mistakes when implementing it, or the complex setup may trigger some bugs that simpler setups do not. There is a real cost to fractions with large denominators. Furthermore, different prime factors are harder or easier to produce: 2 is the easiest, 3 is alright, 5 and up gets nasty.

Therefore, it may be desirable for a production plan to produce slightly less than the maximum sellable goods, in order to have simple denominator fractions, and possibly also produce some non-primary-goal materials for delivery and crafting.

The physical setup needed to produce some difficult fractions can also have some side effects. A notable case study is in one setup for 59/24 multiples of SC Wuling Battery (14.75/min). We produce excess Xircon (a relatively expensive intermediate product), stash it in the depot, and use a depot unloader elsewhere to combine it with Dense Originium Powder to produce batteries. In chemistry terms, the reaction is limited by Dense Originium Powder. This means the depot slowly accumulates Xircon, up to the limit. When the player then goes to do a delivery job, the top up may automatically select Xircon to use, which is undesirable.

To be clear, this Xircon example is an issue of the physical setup - if the Xircon was piped directly to the Packaging Unit, it would simply clog when there is not enough Dense Originium Powder, and exert backpressure on the Xircon production line instead, and it would never fill up in the depot. There is nothing inherently wrong with 59/24. But complex ratios encourage players to seek such solutions which may have unintended side effects.

Taking into account fraction denominators means (at least this part of) the fitness function must be nonlinear.

# Part 5 - Generate alternate solutions

Now we should add a utility function that, possibly taking a random seed, does a search (with some limit on the compute cost) for alternate solutions, and returns the most fit solution it found. The main function should now print, in addition to the top solution found earlier, a most fit solution found by this function.

In terms of how this function would be implemented, it is another design question. Here is one possible way it could be done. Allow 2 types of moves: 

- From the production plan, find a fraction, and round down to a nicer (smaller denominator) fraction or integer.
- Allocate unused inputs to a new or existing material output, producing as much of that output as possible.

And then control the walk with simulated annealing.

You can try implementing multiple backends for this search function (like scipy's optimization functions). Check a sample output. Does it seem qualitatively good? If not, why? If the fitness function doesn't match what we want, you can go back and adjust the fitness function.

It may be possible that a specialized direct solution produces better results than a randomized search. Or maybe a hybrid of backends is best.

Once you have a favourite/best backend, keep that as the default backend to dispatch to.