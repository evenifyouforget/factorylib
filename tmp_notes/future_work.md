To help with future parallel PR drafting, I want to make sure our code here is modular, such that multiple PRs can work on different parts and produce a coherent end product.

In particular, I want to make sure one future PR can be purely concerned with replacing the current simulated annealing with a tuned better alternative. This should be possible as long as our production plan data class supports:

- eq
- hash
- get fitness
- get random new mutant

Currently there is no sense of direction, so gradient descent is not possible. We also can't crossbreed general plans, so no genetic algorithm. But many other algorithms are still possible to try and use.