# Problem statement

We need a function that can solve the following problem:

Suppose we have a set of input resources i in [0, N-1]. Each input resource has A_i supply.

Suppose we have a set of formulas, like 2 A_0 + 5 A_1 = $7. These can be expressed as vectors (together, as a matrix). There are M formulas, numbered j in [0, M-1].

Each formula must be applied with a coefficient c_j in [0, C_j], where C_j is the limit on how many "times" that formula can be used. C_j itself is C_j in (0, infinity]. In code, should use the float constant inf.

Total usage of each input must be less than or equal to supply.

Maximize $ produced (abstract unit of output objective).

Solution should output multiple of each formula, and $ produced.