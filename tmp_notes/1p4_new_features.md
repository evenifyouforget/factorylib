Exact recipes and numbers are on the way, but for now, here's what to expect.

Devs have promised us new, more efficient (less resources, less buildings) ways to produce Xiranite, Heavy Xiranite, and Hetonite. We also have new recipes involving gases. The only logical conclusion is that these new recipes should produce the aforementioned materials more efficiently... but this remains to be proven by the LP solver.

New environment sourced resources:

- Inergen (new gas)
- Xiragen (gaseous form of Xiranite)

There is also a new tier above Hetonite, which is Pyrrolite. Pyrrolite promises a new, cheaper way to craft gear - again, this has yet to be proven by LP, but hopefully the devs didn't lie to us.

# Flow control feature of Item Control Port (belts/solid) and Pipe Control Port (pipe/liquid/gas)

Previously this 1x1 building was only used to filter the item type, or to limit the total count passing through. Now there is a new option, to limit the rate passing through to N/min, where N is an integer in the range [0, 30].

Incidentally, one of the tutorials shows off how useful this is by having us set the flow limit to 6/min. This is relevant later. It also shows off that (at least in this case), 1/5 is no longer a difficult prime number to produce. (a true 1/5 splitter is still hard, and it would come up for 1/25, but that probably won't occur here)

# Threshold activation

Some new recipes require an activation slot.

Rather than consuming the material in a simple ratio, the building will simply consume all of the material fed through the activation slot, and, if it is the correct type and coming in at a rate greater than or equal to the threshold, the recipe is allowed to happen.

To illustrate, consider this fictitious formula:

Electrolysis Unit: 30/min Distilled Water + Electrite (activation, 6/min) -> 30/min Hydrogen + 30/min Oxygen

The following are all valid outcomes from this single building:

30/min Distilled Water + 6/min Electrite -> 30/min Hydrogen + 30/min Oxygen
30/min Distilled Water + 10/min Electrite -> 30/min Hydrogen + 30/min Oxygen
30/min Distilled Water + 30/min Electrite -> 30/min Hydrogen + 30/min Oxygen
1/min Distilled Water + 6/min Electrite -> 1/min Hydrogen + 1/min Oxygen

Indeed, it does not matter if Electrite is 6/min or 30/min - it is fully consumed regardless, and not in any fixed ratio to the actual reactants. But if Electrite is ever less than 6/min, the reaction completely pauses: 0/min Distilled Water -> 0/min Hydrogen + 0/min Oxygen

Perhaps it would be more clear to write it as:

30/min Distilled Water -> 30/min Hydrogen + 30/min Oxygen (under the conditions of 6/min or more Electrite)

Activation would need to be modeled as an integer formula:

6/min Electrite -> 1 Electrite Activation (integer multiples only)

And for each formula that needs it, it needs a distinct counter:

1 Electrite Activation -> 1 Hydrolysis Electrite Activation (integer multiples only)
1 Electrite Activation -> 1 Foobarlysis Electrite Activation (integer multiples only)
30/min Distilled Water + 1 Hydrolysis Electrite Activation -> 30/min Hydrogen + 30/min Oxygen
30/min Foobarium + 1 Foobarlysis Electrite Activation -> 30/min Fooium + 30/min Barium

This is to prevent mistakenly "sharing a building" between 0.5 multiples of Hydrolysis and 0.5 multiples of Foobarlysis, which is physically not possible. You would need separate buildings, and therefore 12/min of Electrite.

Incidentally, the activation energy... er, activation flow, for all recipes, is always 6/min of something. It's never a different number.

# Gas dispersing and gaseous environment

The Gas Dispersing Unit is a 3x3 building that takes 1 gas input. When activated by 6/min of a compatible gas, it creates an "environment" in a 13x13 square centered on itself, of that gas. There are 4 environments:

- Stable (Inergen)
- Humid (Aquagen aka gaseous water)
- Xiranite (Xiragen aka gaseous Xiranite)
- Acrid (Acridgen aka gaseous acid)

You could also say there is a 5th environment, the null environment corresponding to not being inside a gas field.

If a building is fully inside a gas field, it has access to those alternate formulas. For example, some formulas are only possible while a building is fully inside a Stable field.

This square, with the 3x3 cutout in the middle and space for a pipe to reach the Gas Dispersing Unit itself, has just enough space to fit four 5x5 buildings or four 6x4 buildings, which is the largest footprint of building. So assuming the user wants to spend less resources, at the cost of a potentially nastier spatial layout, you can in fact fit 4 environment-enabled recipes per Gas Dispersing Unit. In other words, we can model it like so:

6/min Inergen -> 4 Inergen Environment Allowance (integer multiples only)

And likewise for the other gases.

This could be configurable - perhaps a user who isn't willing to make cursed layouts just to pack 4 buildings inside a gas field, will only fit 3 buildings within a gas field. Though I suspect (this is my personal opinion) that endgame factories will in fact use 4 buildings per gas field.

# Alternate gear crafting

Now, each Component can also be used to craft the lower tier gears. Ex. Hetonite Component can be used to craft gears originally made with Xiranite Component, but not the other way around - Xiranite Component cannot be used to craft gears whose cost is originally in Hetonite Component.

I've nicknamed these Tier 1~4, but that's not an official name.

Interestingly, Pyrrolite Component also comes with a 99% discount on the Wuling Stock Bill part of the cost, meaning it is practically only limited by a user's Component supply, and not their $ income. It seems the devs really want us to make Pyrrolite Component. This also means that making Xiranite Component, Cuprium Component, and Hetonite Component, are completely unnecessary for endgame users now, who can craft all 4 tiers using Pyrrolite Component, and just produce more Pyrrolite Component.

# Kaneko's 6/min Science

These experiments have yet to be done, but Kaneko has pointed out there might be something interesting going on here.

What does it physically mean for something to have a flow of 6/min ?

Let's look at an Inergen source. Like other mineral sources, there are low purity veins (1 unit/6s = 10/min), and high purity veins (1 unit/3s = 20/min). This is true for Inergen even though it's a gas. A user might find 3 high purity veins. They will use a converger to combine the flows. This totals to 60/min. But depending on when the 3 veins are rigged up, they may be in phase or out of phase.

Now suppose this goes into a splitter. One branch has our 6/min flow limited Pipe Control Port. The other branch let's suppose can absorb any amount of flow.

How does the flow control actually work?

There is no continuous flow in Endfield. Under the hood, it is tick based. Let's suppose it's 6 ticks/second. This means 6/min means 1 item every 10 seconds = 60 ticks.

So perhaps what this Pipe Control Port is have a cooldown, where items are rejected unless it has been 60 ticks since the last item.

But remember, the sources may be out of sync rather than a nice stream. There is no guarantee that, after 60 ticks exactly, the next item is ready. Maybe an item was offered by the splitter at tick 58, but rejected, because it's too early, so the item went on the other branch. And then no new item comes until tick 70.

Now suppose downstream we have a threshold reaction, such as a Gas Dispersing Unit. It wants 6/min. What does 6/min mean here? Maybe it means, it has been 60 ticks or less since the previous item.

Oops, our 6/min flow limited Pipe Control Port caused our Gas Dispersing Unit, which wants 6/min, to shut off.

If this happened in reality, players would probably be furious. So there is probably some kind of leniency built in. Maybe the Pipe Control Port itself has a buffer, or the activation energy looks for there to be 10 items in the last 600 ticks, or even 620 ticks.

There is also the natural question of how an artifical 1/5 splitter (see batterylib) would fare. Long term, it does produce a 1/5 ratio, but the items are not evenly spaced.

Besides these questions of how 6/min is implemented, there are also old concerns about the Item Control Port and Pipe Control Port killing offline efficiency. It is known that the offline factory calculation uses a different calculation than the online calculation, which results in belts having a very small speed loss (something like 0.5% loss), which most users are fine with, but most notably, the offline model can't seem to understand Item Control Ports and Pipe Control Ports, and there's a much bigger loss. The old recommendation is that, if you want your factory to run at 100% offline, just don't use the Control Ports. But seeing as the devs are really pushing this 6/min flow limiting feature, they've probably fixed that, or else the players will find out soon enough when they wake up and see that their factory produced nothing while they slept.

Kaneko (and some other Endfield science people) intend to experiment, and find out.
