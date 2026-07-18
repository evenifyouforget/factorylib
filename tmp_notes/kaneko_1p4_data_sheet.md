# Endfield 1.4 Data Sheet (wip)

## Overall

Estimated completion time: 3 days
Total upgrades cost: 7.9M $ (Wuling Stock Bill)
RDM: max is 24, only need 21 for max mineral purity
Elastic goods: max 400, 200/day

## Echoes of War

- Cycle length: 1 week (so do it once a week)
- Level 60: get oroberyls
- Level 90: get Mark of Perseverance
- Bonus: for medal only. Somewhat challenging (maybe comparable to Umbral Monument: Agony difficulty, or CC Re-ignition risk 30)

## Outpost: Cardiac Remediation Station

Max level 3 --> 4
Cost: 1.8M prosperity (about 3 days worth of selling)

## Depot Node: Test Area

Max level 3 --> 4
Cost: 1.2M $
Depot capacity: 14k --> 22k
Pack goods: ? --> 12k mats, 172k $, 145k $ if transferred
(Wuling City pack goods is 9k mats)

## Environment Monitoring: Marker Stone

Max level 3 --> 4
Cost: 500k $

## New AIC Tech Tree

8 Wuling AIC Index -> To access all new recipes
2 Wuling AIC Index -> New turrets (so far not needed for anything)
1 Wuling AIC Index -> Unlock rift neutralization (needed for exploration)
2 Wuling AIC Index -> Convenience features (gas tanks, flow rate control feature)
Total cost: 13 Wuling AIC Index

## Outpost: Cloudseeder Station

Starts with 160k savings
Level 1:

- Sellable goods: Pyrrolite Part (70$), SC Wuling Battery, Heavy Xiranite, Yazhen Syringe A, Jincao Tea, Separator Core (1$), Xiranite
- Need 280k outpost prosperity to level up 1 --> 2 (initial sell + around 1 day for more savings)
- Base earnings: 2700/h
  AIC size upgrade cost: 450k/1.2M
  Depot bus upgrade cost: 300k/800k/1.2M/1.6M

## Redistributor: Yinglung Pass

Level 0 → 1

- Cost: 50k $
- +30 Daily supplies purchase limit
- +15 Daily supplies replenishment limit
  Level 1 → 2
- Cost: 600k $
- +30 Daily supplies purchase limit
- +15 Daily supplies replenishment limit

## Crafting

using Xiranite Component (XC), Cuprium Component (CC), Hetonite Component (HC), or Pyrrolite Component (PC)
Tier 1 gears: 8k $ + 50 XC / 8k $ + 50 CC / 8k $ + 10 HC / 80 $ + 5 PC
Tier 2 gears: 16k $ + 50 CC / 16k $ + 10 HC / 160 $ + 5 PC
Tier 3 gears: 25k $ + 50 HC / 250 $ + 25 PC
Tier 4 gears: 360 $ + 50 PC

## Items needed one-time for mission progress/exploration

Less than 100:

- Separator Core
- Cuprium Canister
- Cuprium Canister filled with Inergen
- Cuprium Canister filled with Xiragen
  15000 or more:
- Liquid Xiranite (for Rift Neutralization)

## Things You May Not Know About The New Buildings

- Gas Dispersing Unit has 3x3 footprint, creates a gas field in 13x13 square centered on itself (so 5 on each side). This is just enough to fit four 5x5 or 6x4 buildings (the largest footprint). A building must be completely within the gas field to benefit from the gas environment.
- Gas Dispersing Unit cannot be placed close together: if their 13x13 squares would overlap, that placement is not possible.
- 6/min threshold activations: it will consume however much is provided, up to 30/min. (if you supply more than 30/min, it results in backpressure)
- Flow Control: Can only be adjusted in increments of 6/min. Range allowed is [0, 30] for Item Control Port (belts), [0, 60] for Pipe Control Port (pipes).
- 6/min threshold activations aren't actually flow based: after receiving 1 item, the building is on for 10 seconds. This means you can actually achieve fractional production with less than 6/min input, however, the interval between items is more important than the long term flow rate.
- 6/min Flow Control can satisfy 6/min activations with 100% uptime\*, both online and offline. \*Occasionally misses a tick offline. See science report.

## New Max Raw Material Income

Originium Ore: 540/min (unchanged)
Ferrium Ore: 120/min (unchanged)
Cuprium Ore: 360/min → 420/min
Inergen: 0/min → at least 460/min
Xiragen: 0/min → 100/min

# New Materials

Inergen (gas)
Aquagen (gaseous form of Water)
Acridgen (gaseous form of Acid)
Xiragen (gaseous form of Xiranite)
Heavy Xiragen (gaseous form of Heavy Xiranite)
Cuprium Gas (gaseous form of Cuprium)
Hetonite Gas (gaseous form of Hetonite)

# New Recipes

Fitting Unit: 5 Pyrrolite → 1 Pyrrolite Part (every 10 seconds)
Moulding Unit: 2 Cuprium + 1 Inergen → 1 Cuprium Canister (every 2 seconds)
Gearing Unit: 1 Pyrrolite + 2 Heavy Xiranite → 1 Pyrrolite Component (every 10 seconds)
Filling Unit: 1 Cuprium Canister + 1 (any gas) → 1 Cuprium Canister filled with that gas (every 2 seconds)
Packaging Unit: 1 Cuprium Canister + 1 Xiranite → 2 Separator Core (every 2 seconds)
Forge of the Sky: 1 Carbon + 1 Water → 1 Xiranite (every 2 seconds, Stable ENV)
Purification Unit: 2 Xiragen + 2 Separator Core → 1 Heavy Xiragen (every 2 seconds)
Purification Unit: 2 Xiragen + 1 Separator Core → 1 Heavy Xiragen (every 2 seconds, Stable ENV)
Purification Unit: 2 Cuprium Gas + 2 Separator Core → 1 Hetonite Gas (every 2 seconds)
Purification Unit: 2 Cuprium Gas + 1 Separator Core → 1 Hetonite Gas (every 2 seconds, Stable ENV)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 1 Water → 1 Aquagen (every 2 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 1 Liquid Xiranite → 1 Xiragen (every 2 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 2 Cuprium Solution → 1 Cuprium Gas (every 2 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 1 Acid → 1 Acridgen (every 2 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 2 Liquid Heavy Xiranite → 5 Heavy Xiragen (every 10 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + 1 Hetonite Solution → 1 Hetonite Gas (every 2 seconds)
Fluid-Gas Transmuting Unit: Liquid Xiranite [threshold 6/min] + (X units of the gas) → (Y units of the liquid) (every Z seconds) (reverse of the above 6 recipes)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + 1 Xiranite → 1 Xiragen (every 2 seconds)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + 2 Heavy Xiranite → 5 Heavy Xiragen (every 10 seconds)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + 2 Cuprium → 1 Cuprium Gas (every 2 seconds)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + 1 Hetonite → 2 Hetonite Gas (every 2 seconds)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + 1 Pyrrolite → 1 Pyrrolite Gas (every 2 seconds)
Solid-Gas Transmuting Unit: Xiragen [threshold 6/min] + (X units of the gas) → (Y units of the solid) (every Z seconds) (reverse of the above 5 recipes)
Gas Reactor Globe: 2 Hetonite Gas + 1 Xiragen → 1 Pyrrolite Gas (every 2 seconds, Acrid ENV)

## Gas Dispersing Unit

Note these are special: they create the 13x13 gaseous environment (ENV) rather than outputting a material, and some recipes only work in a specific environment.
Gas Dispersing Unit: Inergen [threshold 6/min] → Stable ENV
Gas Dispersing Unit: Aquagen [threshold 6/min] → Humid ENV
Gas Dispersing Unit: Acridgen [threshold 6/min] → Acrid ENV
Gas Dispersing Unit: Xiragen [threshold 6/min] → Xiranite ENV

## Other existing formulas that I maybe forgot to document

Reactor Crucible: 1 Xiranite + 1 Water → 1 Liquid Xiranite (every 2 seconds)
Reactor Crucible: 1 Heavy Xiranite + 1 Acid → 1 Liquid Heavy Xiranite (every 2 seconds)
Water Treatment Unit: 1 Sewage → (nothing) (every 2 seconds)
Water Treatment Unit: 1 Xircon Effluent → (nothing) (every 2 seconds)
Water Treatment Unit: 1 Inert Xircon Effluent → (nothing) (every 2 seconds)

# Threshold Activations Science Report

Experimental setup:
6/min Pipe Control Port → Gas Dispersing Unit
4 Forge of the Sky producing Xiranite → gap detection circuit
All FotS placed manually, so the phase offset relative to each other and the Gas Dispersing Unit is effectively random.
Note the formula here: 1 Carbon + 1 Water + (Stable ENV) → 1 Xiranite, every 2 seconds. Each FotS has its own internal clock (phase), where every 2 seconds, it attempts to produce an item, and if it fails to produce an item, it waits 2 seconds to try again.
Logged off for about 14 minutes, returned.
1 of the 4 production lines detected a gap. Meaning the other 3 were producing 30/min Xiranite at 100% efficiency, while the bad 1 failed to produce an item in some 2 second interval.
How could this possibly happen?
The Pipe Control Port’s flow control seems generally consistent - for it to deliver 1 Inergen every 10 seconds, it probably has its own internal buffer, so if it is offered 1 Inergen at 9 s and 11 s, rather than rejecting the Inergen at 9 s (which results in 1 s of downtime), it takes the Inergen at 9 s, holds onto it, and releases it at 10 s. This is good news - it means the 6/min Pipe Control Port is mostly reliable even for offline factories, for its intended use of supplying a 6/min threshold formula.
If the Pipe Control Port operated on a strict 2 second cycle, if it could not produce an item at 10 s, it would wait until 12 s to try again. In this case, all 4 FotS would miss a production opportunity, but that’s not what we observe.
So we must conclude that the downtime of the Gas Dispersing Unit was less than 2 seconds, and this downtime only coincided with one of the FotS’ production opportunities, but not the other three FotS. This also means the Gas Dispersing Unit and Pipe Control Port either have a flexible schedule, or have an internal clock which is faster than every 2 seconds - perhaps every 1 second, or every 0.5 seconds.
