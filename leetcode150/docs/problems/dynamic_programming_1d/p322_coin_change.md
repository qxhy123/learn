# 322. Coin Change

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/coin-change/
- Official Group: 1D DP
- Pattern Group: Dynamic Programming 1D
- Patterns: dynamic-programming-1d

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given:

```text
coins  = denominations of coins you may use
amount = target amount of money
```

Return the fewest number of coins needed to make exactly `amount`.

You may use each coin denomination unlimited times.

If it is impossible to make exactly `amount`, return `-1`.

For example:

```text
coins = [1, 2, 5]
amount = 11
```

One valid way to make `11` is:

```text
5 + 5 + 1 = 11
```

That uses `3` coins.

Another valid way is:

```text
5 + 2 + 2 + 2 = 11
```

That uses `4` coins.

The problem does not ask whether an amount can be made. It asks for the **minimum number of coins** among all ways to make it.

So the real problem is:

> Among all multisets of the given coin values whose sum is exactly `amount`, find the one with the smallest number of coins.

Two details are important:

1. Coin order does not matter to the final answer.
2. Coin reuse is unlimited.

So these are the same solution:

```text
5 + 5 + 1
5 + 1 + 5
1 + 5 + 5
```

They all use the same `3` coins and produce the same amount.

---

### 2. Start From the Brute-Force Recursion

A first direct way to think about the problem is:

> To make an amount, choose the first coin, then solve the remaining amount.

If we choose coin `c`, then the remaining amount becomes:

```text
amount - c
```

If the best way to make `amount - c` uses `k` coins, then the best way using `c` as the current coin uses:

```text
1 + k
```

So we can try every possible first coin.

Conceptually:

```python
def solve(remaining):
    if remaining == 0:
        return 0

    if remaining < 0:
        return infinity

    best = infinity

    for coin in coins:
        candidate = 1 + solve(remaining - coin)
        best = min(best, candidate)

    return best
```

Then the final answer is:

```python
answer = solve(amount)
```

If `answer` is still infinity, return `-1`.

This recursion is correct as a search idea because every valid coin combination has a last recursive path made of individual coin choices.

For `coins = [1, 2, 5]` and `amount = 11`, the recursion starts like this:

```text
solve(11)
  try 1 -> 1 + solve(10)
  try 2 -> 1 + solve(9)
  try 5 -> 1 + solve(6)
```

Then `solve(10)` branches again:

```text
solve(10)
  try 1 -> 1 + solve(9)
  try 2 -> 1 + solve(8)
  try 5 -> 1 + solve(5)
```

Notice the repeated subproblem:

```text
solve(9)
```

It appears from more than one path.

That is the central inefficiency.

The brute-force recursion repeatedly asks the same question:

```text
What is the fewest number of coins needed to make amount x?
```

for the same values of `x`.

---

### 3. The Key Observation

The remaining amount fully determines the future.

If two recursive paths both reach:

```text
remaining = 9
```

then it no longer matters how they got there.

The question is exactly the same:

```text
What is the minimum number of coins needed to make 9?
```

The previous choices may affect how many coins have already been used, but the optimal cost from this point forward depends only on `9`, not on the path.

That gives us overlapping subproblems:

```text
solve(9), solve(8), solve(7), ...
```

The problem also has optimal substructure:

```text
best answer for amount x
= 1 coin just chosen + best answer for amount x - chosen_coin
```

So instead of recomputing `solve(x)` many times, compute the answer for each amount once.

There are only:

```text
0, 1, 2, ..., amount
```

possible remaining target values.

That is why this is a one-dimensional dynamic programming problem.

---

### 4. DP State and Invariant

Define:

```text
dp[x] = the fewest number of coins needed to make exactly amount x
```

If `x` cannot be made using the given coins, then:

```text
dp[x] = infinity
```

The base case is:

```text
dp[0] = 0
```

Why?

Because making amount `0` requires no coins.

This is not just a convenient initialization. It is the anchor that allows every exact coin value to be formed:

```text
if coin = 5:
    dp[5] can become 1 + dp[0]
```

The invariant we want after processing amount `x` is:

```text
dp[x] is the minimum number of coins needed to make exactly x,
or infinity if x is impossible.
```

The final answer comes from:

```text
dp[amount]
```

If it is infinity, return `-1`.

---

### 5. Deriving the Transition From the Last Coin

To compute `dp[x]`, ask:

> What could the last coin be in an optimal solution for amount `x`?

Suppose the last coin is `coin`.

Then before taking that last coin, we must have made:

```text
x - coin
```

If `x - coin` is possible, then one candidate solution for `x` is:

```text
dp[x - coin] + 1
```

The `+ 1` counts the last coin.

Since the last coin could be any denomination in `coins`, try them all:

```text
dp[x] = min(dp[x - coin] + 1) over every coin where x - coin >= 0
```

If no coin leads from a reachable smaller amount, `dp[x]` remains infinity.

This transition is the whole algorithm.

The important part is that each transition moves from a smaller amount to a larger amount:

```text
x - coin < x
```

because coin values are positive.

So if we fill `dp` from `0` upward to `amount`, every dependency is already known when we need it.

---

### 6. Detailed Algorithm

1. Create an array of length `amount + 1`.
2. Fill it with a sentinel value meaning impossible.
3. Set `dp[0] = 0`.
4. For every target value `x` from `1` to `amount`:
   - Try every coin denomination.
   - If `coin <= x`, look at `dp[x - coin]`.
   - If `x - coin` is reachable, update `dp[x]` with `dp[x - coin] + 1`.
5. After all amounts are processed:
   - If `dp[amount]` is still impossible, return `-1`.
   - Otherwise return `dp[amount]`.

A practical sentinel is:

```text
amount + 1
```

Why is that safe?

If the answer exists, it never needs more than `amount` coins when coin `1` exists, because `1 + 1 + ... + 1` uses exactly `amount` coins.

If coin `1` does not exist, the answer is either smaller than `amount` or impossible.

So `amount + 1` is larger than any meaningful valid answer.

---

### 7. Pseudocode

```python
def coinChange(coins, amount):
    impossible = amount + 1
    dp = [impossible] * (amount + 1)
    dp[0] = 0

    for x in range(1, amount + 1):
        for coin in coins:
            if coin <= x:
                dp[x] = min(dp[x], dp[x - coin] + 1)

    if dp[amount] == impossible:
        return -1

    return dp[amount]
```

The same idea can also be written as a top-down memoized recursion:

```python
def coinChange(coins, amount):
    memo = {}

    def solve(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return infinity
        if remaining in memo:
            return memo[remaining]

        best = infinity
        for coin in coins:
            best = min(best, 1 + solve(remaining - coin))

        memo[remaining] = best
        return best

    answer = solve(amount)
    return -1 if answer == infinity else answer
```

Both versions compute the same states.

The bottom-up version is often simpler because the fill order is explicit:

```text
0 -> 1 -> 2 -> ... -> amount
```

---

### 8. Detailed Example Walkthrough

Use:

```text
coins = [1, 2, 5]
amount = 11
```

Initialize:

```text
dp[0] = 0
all other dp[x] = infinity
```

So conceptually:

```text
amount: 0  1  2  3  4  5  6  7  8  9  10 11
dp:     0  ∞  ∞  ∞  ∞  ∞  ∞  ∞  ∞  ∞  ∞  ∞
```

Now fill amounts from left to right.

For `x = 1`:

```text
coin 1 works: dp[1] = dp[0] + 1 = 1
coin 2 too large
coin 5 too large
```

```text
dp[1] = 1
```

For `x = 2`:

```text
coin 1: dp[1] + 1 = 2
coin 2: dp[0] + 1 = 1
coin 5: too large
```

Best is `1`, using coin `2`.

```text
dp[2] = 1
```

For `x = 3`:

```text
coin 1: dp[2] + 1 = 2     (2 + 1)
coin 2: dp[1] + 1 = 2     (1 + 2)
coin 5: too large
```

```text
dp[3] = 2
```

For `x = 4`:

```text
coin 1: dp[3] + 1 = 3
coin 2: dp[2] + 1 = 2
coin 5: too large
```

```text
dp[4] = 2
```

For `x = 5`:

```text
coin 1: dp[4] + 1 = 3
coin 2: dp[3] + 1 = 3
coin 5: dp[0] + 1 = 1
```

```text
dp[5] = 1
```

The direct coin `5` is best.

Continue the same process:

```text
x = 6:
  coin 1 -> dp[5] + 1 = 2   (5 + 1)
  coin 2 -> dp[4] + 1 = 3
  coin 5 -> dp[1] + 1 = 2   (1 + 5)
  dp[6] = 2

x = 7:
  coin 1 -> dp[6] + 1 = 3
  coin 2 -> dp[5] + 1 = 2   (5 + 2)
  coin 5 -> dp[2] + 1 = 2   (2 + 5)
  dp[7] = 2

x = 8:
  coin 1 -> dp[7] + 1 = 3
  coin 2 -> dp[6] + 1 = 3
  coin 5 -> dp[3] + 1 = 3
  dp[8] = 3

x = 9:
  coin 1 -> dp[8] + 1 = 4
  coin 2 -> dp[7] + 1 = 3
  coin 5 -> dp[4] + 1 = 3
  dp[9] = 3

x = 10:
  coin 1 -> dp[9] + 1 = 4
  coin 2 -> dp[8] + 1 = 4
  coin 5 -> dp[5] + 1 = 2   (5 + 5)
  dp[10] = 2

x = 11:
  coin 1 -> dp[10] + 1 = 3  (5 + 5 + 1)
  coin 2 -> dp[9] + 1 = 4
  coin 5 -> dp[6] + 1 = 3   (5 + 1 + 5)
  dp[11] = 3
```

Final table:

```text
amount: 0  1  2  3  4  5  6  7  8  9  10 11
dp:     0  1  1  2  2  1  2  2  3  3  2  3
```

So the answer is:

```text
3
```

One optimal construction is:

```text
5 + 5 + 1
```

---

### 9. Impossible Amount Walkthrough

Use:

```text
coins = [2]
amount = 3
```

Initialize:

```text
dp[0] = 0
```

For `x = 1`:

```text
coin 2 is too large
```

So:

```text
dp[1] = infinity
```

For `x = 2`:

```text
coin 2: dp[0] + 1 = 1
```

So:

```text
dp[2] = 1
```

For `x = 3`:

```text
coin 2: dp[1] + 1
```

But `dp[1]` is infinity because amount `1` cannot be made.

So `dp[3]` remains infinity.

Final result:

```text
-1
```

This shows why the sentinel is necessary: some amounts have no exact representation.

---

### 10. Why Greedy Does Not Work

A tempting idea is:

> Always take the largest coin that does not exceed the remaining amount.

That works for some coin systems, including many real-world currencies, but it is not guaranteed here.

Example:

```text
coins = [1, 3, 4]
amount = 6
```

Greedy takes:

```text
4 + 1 + 1 = 6
```

That uses `3` coins.

But the optimal answer is:

```text
3 + 3 = 6
```

That uses `2` coins.

The failure happens because choosing the locally largest coin can leave a bad remainder.

Dynamic programming avoids this by considering every possible last coin for every amount.

---

### 11. Correctness

We prove the bottom-up algorithm returns the minimum number of coins needed to make `amount`, or `-1` if impossible.

#### State Meaning

For every `x` from `0` to `amount`, `dp[x]` is intended to store the minimum number of coins needed to make exactly `x`, or the sentinel value if `x` is impossible.

#### Base Case

```text
dp[0] = 0
```

This is correct because amount `0` can be made using zero coins, and no solution can use fewer than zero coins.

All other states initially hold the sentinel, meaning no construction has been found yet.

#### Inductive Step

Assume that for every smaller amount `< x`, the table already stores the correct minimum number of coins or correctly marks the amount impossible.

To compute `dp[x]`, the algorithm tries every coin denomination `coin` with `coin <= x`.

If an optimal solution for `x` ends with `coin`, then the coins before that last coin must make exactly:

```text
x - coin
```

By the induction assumption, `dp[x - coin]` already contains the fewest coins needed for that smaller amount.

So the best solution for `x` whose last coin is `coin` has size:

```text
dp[x - coin] + 1
```

The algorithm takes the minimum over all possible last coins.

Therefore it considers every possible form of an optimal solution for `x`, because every non-empty coin solution has some last coin.

If none of those smaller states are reachable, then no last coin can complete amount `x`, so `x` is impossible and the sentinel remains correct.

Thus `dp[x]` is correct.

#### Conclusion

By induction from `0` through `amount`, every `dp[x]` is correct after it is filled.

Therefore `dp[amount]` is correct.

The algorithm returns `dp[amount]` when reachable and `-1` when it remains impossible, exactly matching the problem requirement.

---

### 12. Complexity

Let:

```text
A = amount
C = number of coin denominations
```

For each amount from `1` to `A`, the algorithm tries each coin.

So the time complexity is:

```text
O(A * C)
```

The DP table has one entry for each amount from `0` through `A`.

So the space complexity is:

```text
O(A)
```

This space is not reducible to constant space in the usual bottom-up form, because a state `dp[x]` may depend on many earlier amounts:

```text
dp[x - coin]
```

for every coin denomination.

---

### 13. Common Pitfalls

#### Pitfall 1: Returning the sentinel instead of `-1`

The DP table may use `amount + 1` or infinity internally.

But the problem requires:

```text
-1
```

when the target cannot be made.

Always convert the sentinel at the end.

#### Pitfall 2: Forgetting the `amount = 0` case

If `amount` is `0`, the answer is immediately:

```text
0
```

The standard DP handles this naturally because `dp[0] = 0` and the loops do not need to improve it.

#### Pitfall 3: Treating coin order as meaningful

The answer depends only on the number of coins, not on the order in which they are chosen.

The DP transition may describe a "last coin," but that is just a proof technique. It does not mean the final combination has an important order.

#### Pitfall 4: Using greedy because it feels natural

Choosing the largest possible coin can fail for arbitrary denominations.

Use DP unless the problem explicitly guarantees a coin system where greedy is valid.

#### Pitfall 5: Confusing this with counting combinations

This problem asks:

```text
minimum number of coins
```

It does not ask:

```text
number of ways to make the amount
```

So the transition uses `min`, not addition of counts.

Compare:

```text
coin change minimum: dp[x] = min(dp[x], dp[x - coin] + 1)
coin change counting: dp[x] += dp[x - coin]
```

Those are different problems.

#### Pitfall 6: Letting impossible states create fake answers

If `dp[x - coin]` is infinity, then:

```text
dp[x - coin] + 1
```

is still impossible.

Using `amount + 1` as the sentinel makes this safe in Python, but conceptually you should remember that unreachable states must not become real candidates.

---

### 14. First-Principles Summary

The problem asks for the smallest number of reusable coins whose values sum exactly to `amount`.

The brute-force idea chooses a coin and recursively solves the remaining amount.

That recursion repeats the same subproblems because many different coin sequences can lead to the same remaining amount.

The remaining amount is the only information needed to solve the rest of the problem, so define:

```text
dp[x] = minimum coins needed to make x
```

The base case is:

```text
dp[0] = 0
```

For every amount `x`, derive the answer by considering the last coin used:

```text
dp[x] = min(dp[x - coin] + 1)
```

over all valid coins.

Filling amounts from small to large guarantees that every smaller amount has already been solved.

At the end, `dp[amount]` is either the optimal coin count or still impossible, in which case the required output is `-1`.

## Implementation

See `solutions/dynamic_programming_1d/p322_coin_change.py`.

## Tests

See `tests/dynamic_programming_1d/test_p322_coin_change.py`.

## Examples

### Example 1

- Input: `{'coins': [1, 2, 5], 'amount': 11}`
- Output: `3`

### Example 2

- Input: `{'coins': [2], 'amount': 3}`
- Output: `-1`

### Example 3

- Input: `{'coins': [1], 'amount': 0}`
- Output: `0`

## Follow-up Practice

- Write the state definition before code: what does `dp[x]` mean exactly?
- Derive the transition by asking what the last coin could be.
- Implement both top-down memoization and bottom-up tabulation.
- Test an impossible amount such as `coins = [2]`, `amount = 3`.
- Test a case where greedy fails, such as `coins = [1, 3, 4]`, `amount = 6`.
