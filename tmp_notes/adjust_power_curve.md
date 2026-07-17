TLDR power should reward much less for producing extra, and the hard cap can be lowered to 110%.

Why:

There are two regimes:

- Average power based: In this case, the energy reserves do not over-charge and waste energy during the charging part of the cycle.
- Drain and fill based: In this case, energy reserves do over-charge and waste energy during the charging part of the cycle.

Consider a power draw of 6600 + epsilon W. 6600 W is covered by the base 200 W, plus 2x 3200 W from SC Wuling Battery. But the small epsilon slowly drains reserves. We have to eventually feed 1 more SC Wuling Battery.

This produces 3200 W x 40 s = 128000 J, which is more than the 100000 J of energy reserves. 1-100/128 = 21.875% of the energy produced by this battery is wasted!

This is less bad than it looks when we're already in the ballpark of 6600 W. For most 40s intervals, we're already spending 2 batteries. For one interval, we spend 3 batteries, and a fraction of 1 battery is wasted.

Within this regime, from 6600 W to 6600 W + 700 W, if you find the maximum for power wasting, it should be less than 1%.

As for the other regime, a theoretically perfect balancer wastes 0 power.

The actual loss is more likely to come from usage of a balancer that uses a simpler fraction than the exact power demand. Empirically, DIGE's 5% is good enough for most possible power demands. 10% should allow all users to use relatively simple fractions.

There is no need for 40% excess power.