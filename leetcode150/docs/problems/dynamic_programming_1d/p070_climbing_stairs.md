# 70. Climbing Stairs

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/climbing-stairs/
- Official Group: 1D DP
- Pattern Group: Dynamic Programming 1D
- Patterns: dynamic-programming-1d

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are standing at the bottom of a staircase with `n` steps.

Each move can climb either:

```text
1 step
or
2 steps
```

The question is:

> How many different sequences of 1-step and 2-step moves land exactly on step `n`?

For example, if `n = 3`, the valid ways are:

```text
1 + 1 + 1
1 + 2
2 + 1
```

So the answer is `3`.

This is not asking for the minimum number of moves. It is not asking whether the top is reachable. Since every `n >= 1` is reachable using only 1-step moves, reachability is trivial.

The real problem is counting:

```text
number of valid move sequences whose sum is exactly n
```

Order matters.

For `n = 3`:

```text
1 + 2
```

and:

```text
2 + 1
```

are different ways, because the moves happen in a different order.

---

### 2. Start From the Brute Force Recursion

A direct first idea is to build every possible path.

From step `0`, choose either:

```text
take 1 step
take 2 steps
```

Then repeat until you either land exactly on step `n` or go past it.

Conceptually:

```python
def count_from(step):
    if step == n:
        return 1
    if step > n:
        return 0

    return count_from(step + 1) + count_from(step + 2)
```

This is correct because every valid path from `step` must begin with exactly one of two possible moves:

```text
1-step move
2-step move
```

Those two choices are disjoint. A path cannot start with both. So the total number of paths is the sum of the number of paths after each choice.

But the recursion repeats work.

For `n = 5`, the recursion tree includes `count_from(3)` multiple times:

```text
count_from(0)
├── count_from(1)
│   ├── count_from(2)
│   │   ├── count_from(3)
│   │   └── count_from(4)
│   └── count_from(3)
└── count_from(2)
    ├── count_from(3)
    └── count_from(4)
```

The question "how many ways are there from step `3` to the top?" has the same answer every time it appears. Recomputing it is wasteful.

That repeated-subproblem structure is the reason dynamic programming applies.

---

### 3. Key Observation: Every Path Has a Last Move

Instead of thinking forward from the ground, think backward from the top.

Any valid sequence that lands on step `i` must have ended in exactly one of two ways:

```text
It came from step i - 1 using a 1-step move.
It came from step i - 2 using a 2-step move.
```

There is no third possibility, because the only allowed move sizes are `1` and `2`.

So if we know:

```text
ways to reach step i - 1
ways to reach step i - 2
```

then we can compute:

```text
ways to reach step i
```

by adding them:

```text
ways(i) = ways(i - 1) + ways(i - 2)
```

Why addition?

Because the two groups are disjoint:

- paths whose last move is `1`
- paths whose last move is `2`

No path can have both last moves at the same time.

This is the entire problem-specific recurrence.

---

### 4. DP State and Invariant

Define:

```text
dp[i] = number of different move sequences that land exactly on step i
```

The invariant while filling the table is:

```text
After computing dp[i], dp[i] is exactly the number of valid sequences whose moves sum to i.
```

This state is precise because the only information needed to extend a path is the step it currently reaches.

To compute `dp[i]`, use the last move:

```text
last move was 1 step -> previous position was i - 1 -> dp[i - 1] ways
last move was 2 steps -> previous position was i - 2 -> dp[i - 2] ways
```

Therefore:

```text
dp[i] = dp[i - 1] + dp[i - 2]
```

The base cases come from the smallest stair counts.

For `0` steps, there is one empty way:

```text
do nothing
```

So:

```text
dp[0] = 1
```

For `1` step, there is one way:

```text
1
```

So:

```text
dp[1] = 1
```

Then:

```text
dp[2] = dp[1] + dp[0] = 1 + 1 = 2
```

corresponding to:

```text
1 + 1
2
```

Some implementations instead use:

```text
dp[1] = 1
dp[2] = 2
```

and start the loop at `3`. Both definitions are equivalent for this problem. The `dp[0] = 1` version is often easier to reason about from the recurrence because it treats the final 2-step jump from ground to step `2` naturally.

---

### 5. Detailed Algorithm

Use bottom-up dynamic programming:

1. If `n` is `0`, return `1` under the mathematical empty-path definition. On LeetCode, `n` is at least `1`, so this case usually does not appear.
2. Create an array `dp` of length `n + 1`.
3. Set:

```text
dp[0] = 1
dp[1] = 1
```

4. For each step `i` from `2` through `n`, compute:

```text
dp[i] = dp[i - 1] + dp[i - 2]
```

5. Return `dp[n]`.

Because each state depends only on the two previous states, the array can be reduced to two variables:

```text
previous_two = ways to reach i - 2
previous_one = ways to reach i - 1
current      = ways to reach i
```

For each new step:

```text
current = previous_one + previous_two
```

Then slide the variables forward:

```text
previous_two = previous_one
previous_one = current
```

This space-optimized version preserves the same invariant, just without storing every older state.

---

### 6. Example Walkthrough: n = 5

We want the number of move sequences that sum to `5`.

Start with the base cases:

```text
dp[0] = 1
dp[1] = 1
```

Now fill the table.

For step `2`:

```text
dp[2] = dp[1] + dp[0]
      = 1 + 1
      = 2
```

The ways are:

```text
1 + 1
2
```

For step `3`:

```text
dp[3] = dp[2] + dp[1]
      = 2 + 1
      = 3
```

The ways are:

```text
1 + 1 + 1
1 + 2
2 + 1
```

For step `4`:

```text
dp[4] = dp[3] + dp[2]
      = 3 + 2
      = 5
```

The ways can be grouped by last move:

```text
last move is 1:
  ways to reach 3, then add 1

last move is 2:
  ways to reach 2, then add 2
```

So there are:

```text
3 + 2 = 5
```

For step `5`:

```text
dp[5] = dp[4] + dp[3]
      = 5 + 3
      = 8
```

The final answer is:

```text
8
```

The table is:

```text
i:     0  1  2  3  4  5
dp[i]: 1  1  2  3  5  8
```

---

### 7. Code / Pseudocode

Array-based version:

```python
def climb_stairs(n: int) -> int:
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
```

Space-optimized version:

```python
def climb_stairs(n: int) -> int:
    if n <= 1:
        return 1

    two_steps_before = 1  # ways to reach step 0
    one_step_before = 1   # ways to reach step 1

    for step in range(2, n + 1):
        current = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = current

    return one_step_before
```

If you prefer the LeetCode-facing base cases, the same idea can be written as:

```python
def climb_stairs(n: int) -> int:
    if n <= 2:
        return n

    ways_to_previous_two = 1  # step 1
    ways_to_previous_one = 2  # step 2

    for step in range(3, n + 1):
        ways_to_current = ways_to_previous_one + ways_to_previous_two
        ways_to_previous_two = ways_to_previous_one
        ways_to_previous_one = ways_to_current

    return ways_to_previous_one
```

Both versions compute the same recurrence. They only choose different starting points.

---

### 8. Correctness

We prove that the algorithm returns the number of valid ways to climb exactly `n` steps.

Define `dp[i]` as the number of valid move sequences that land exactly on step `i`.

Base cases:

- `dp[0] = 1` is correct because there is exactly one sequence that reaches step `0`: the empty sequence.
- `dp[1] = 1` is correct because there is exactly one sequence that reaches step `1`: take one 1-step move.

Inductive step:

Assume `dp[i - 1]` and `dp[i - 2]` are correct for some `i >= 2`.

Every valid sequence that reaches step `i` has a final move. Since the only allowed move sizes are `1` and `2`, that final move must be one of these two cases:

```text
from i - 1 to i
from i - 2 to i
```

The number of sequences in the first case is exactly `dp[i - 1]`, because each valid sequence to `i - 1` becomes a unique valid sequence to `i` by appending a 1-step move.

The number of sequences in the second case is exactly `dp[i - 2]`, because each valid sequence to `i - 2` becomes a unique valid sequence to `i` by appending a 2-step move.

These two cases do not overlap because a sequence has exactly one final move. Therefore the total number of valid sequences to `i` is:

```text
dp[i - 1] + dp[i - 2]
```

which is exactly how the algorithm computes `dp[i]`.

By induction, every computed `dp[i]` is correct, including `dp[n]`. Therefore the algorithm returns the correct answer.

The space-optimized version is correct for the same reason: before each iteration, its two variables hold the two DP states needed by the recurrence, and after the update they hold the next pair of states.

---

### 9. Complexity

Let `n` be the number of stairs.

Array-based DP:

- Time: `O(n)` because each step from `2` to `n` is computed once.
- Space: `O(n)` for the DP table.

Space-optimized DP:

- Time: `O(n)` for the same reason.
- Space: `O(1)` because only two previous values are stored.

---

### 10. Common Pitfalls

- Treating `1 + 2` and `2 + 1` as the same. They are different sequences because order matters.
- Returning the minimum number of moves instead of the number of possible move sequences.
- Forgetting the `n = 1` case when using `dp[1] = 1` and `dp[2] = 2`.
- Setting `dp[0] = 0` in the recurrence-based version. That would make `dp[2]` incorrectly become `1` instead of `2`, because it loses the direct `2`-step jump from the ground.
- Using plain recursion without memoization. It is logically correct but recomputes the same subproblems exponentially many times.
- Updating the two space-optimized variables in the wrong order. Compute `current` first, then shift the older values forward.

---

### 11. First-Principles Summary

The staircase problem is a counting problem over sequences of moves.

The brute-force recursion branches on the next move:

```text
1 step or 2 steps
```

But many branches ask the same question again:

```text
How many ways are there to finish from this step?
```

Dynamic programming removes that repeated work by defining a state:

```text
dp[i] = number of ways to reach step i
```

The recurrence comes from the last move, not from a generic template:

```text
To reach i, the path must come from i - 1 or i - 2.
```

So:

```text
dp[i] = dp[i - 1] + dp[i - 2]
```

Once that invariant is clear, the algorithm is just filling the steps in increasing order, or keeping the last two values because no older values are needed.

## Implementation
See `solutions/dynamic_programming_1d/p070_climbing_stairs.py`.

## Tests
See `tests/dynamic_programming_1d/test_p070_climbing_stairs.py`.

## Examples

### Example 1
- Input: `{'n': 2}`
- Output: `2`

### Example 2
- Input: `{'n': 3}`
- Output: `3`

## Follow-up Practice
- Write both the forward recursion and the backward last-move recurrence, then compare where repeated work appears.
- Implement the memoized recursive version before the bottom-up version.
- Explain why the answer follows the Fibonacci recurrence but should still be derived from the staircase moves.
