After you are done with functional correctness, try running the CLI. Are all goals met?

User also has some further questions.

# 1. Is Pyrrolite Part the most efficient sellable good?

It has the highest price per unit. One might think it's intended to be the best sellable good. Does LP choose it? If not, why is Pyrrolite Part not chosen?

If it's because Hetonite Part is currently incentivized to break ties - try incentivizing Pyrrolite Part instead.

# 2. Which constants can be tightened?

When a guidemaker seeks to make a factory guide, they probably want it to be easy to follow. So they will bake in additional safety.

Which of these changes are "safe"? (all goals can still be met, no loss in production) And to what extent?

- Disable purification node
- Increase $/min sellable goods goal
- Increase W power goal
- Decrease base Originium Ore/Ferrium Ore/Cuprium Ore/Inergen/Xiragen
- Increase delivery jobs goal

# 3. Does adding randomness to the delivery job simulator result in a meaningfully different outcome?

Let's say the user logs in at a random time each day. If the random factor is r in [0, 1], then the time that user logs in on day N is at N + r \* random.uniform(0, 1). This means some days have a longer time since the last day, and some days have a shorter time.

It is probably realistic to have some variation here. We could set r = 0.5 by default.
