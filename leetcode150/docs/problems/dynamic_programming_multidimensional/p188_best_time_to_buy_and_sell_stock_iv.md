# 188. Best Time to Buy and Sell Stock IV

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
k      = the maximum number of transactions allowed
prices = prices[i] is the stock price on day i
```

A transaction is:

```text
buy once, then later sell once
```

You may not hold more than one share at a time. That means you cannot buy again before selling the stock you already hold.

The task is to return the maximum profit possible using at most `k` transactions.

For example:

```text
k = 2
prices = [3, 2, 6, 5, 0, 3]
```

One optimal plan is:

```text
buy at 2, sell at 6  -> profit 4
buy at 0, sell at 3  -> profit 3
```

Total profit:

```text
4 + 3 = 7
```

So the answer is `7`.

The real problem is:

> Choose up to `k` non-overlapping buy/sell pairs, in chronological order, so that the sum of their gains is as large as possible.

---

### 2. The Rules That Shape the Problem

There are three important constraints hidden in the wording.

First, a transaction must be completed in order:

```text
buy before sell
```

You cannot sell before you buy.

Second, transactions cannot overlap:

```text
invalid: buy, buy, sell, sell
valid:   buy, sell, buy, sell
```

So every day, you are in exactly one of two physical states:

```text
not holding a stock
holding one stock
```

Third, the limit is on completed transactions. A transaction is counted when a sell happens, because only then have you finished a buy/sell pair.

This suggests that the algorithm must remember more than the day number. It must also remember:

```text
how many transactions have been completed
whether we currently hold a stock
```

That is why this is a multidimensional dynamic programming problem.

---

### 3. Start From the Brute Force Baseline

The most direct idea is to try every legal sequence of actions.

On each day, we could choose one of these actions:

```text
if not holding: skip or buy
if holding:     skip or sell
```

A recursive brute force could look like this:

```python
def dfs(day, transactions_done, holding):
    if day == len(prices):
        return 0

    best = dfs(day + 1, transactions_done, holding)  # skip

    if holding:
        # sell today; this completes one transaction
        if transactions_done < k:
            best = max(
                best,
                prices[day] + dfs(day + 1, transactions_done + 1, False),
            )
    else:
        # buy today; profit decreases by the price paid
        best = max(
            best,
            -prices[day] + dfs(day + 1, transactions_done, True),
        )

    return best
```

This expresses the problem correctly, but without memoization it branches repeatedly and can become exponential.

The same subproblem appears many times. For example:

```text
(day = 4, transactions_done = 1, holding = False)
```

can be reached by many different earlier buy/sell choices. Once we reach that same state, the best possible future profit is identical no matter how we got there.

So the first-principles improvement is:

> If the future only depends on `(day, completed transactions, holding state)`, cache or tabulate that state.

---

### 4. Why a One-Dimensional State Is Not Enough

For easier stock problems, it may be enough to track one best buy price or one best profit.

Here, that loses essential information.

Suppose we know only:

```text
best profit so far
```

That is not enough, because a profit of `4` after one completed transaction is different from a profit of `4` after two completed transactions. The first state can still make another transaction if `k = 2`; the second cannot.

Suppose we know only:

```text
best profit after t transactions
```

That is still not enough, because we also need to know whether that profit is represented while holding a stock or while not holding one.

These two situations are fundamentally different:

```text
not holding: profit is already realized and can be returned as an answer
holding:     profit includes the cost of an open buy and cannot be final yet
```

So a complete state needs both dimensions:

```text
transaction count
holding status
```

The day dimension can be processed by scanning prices from left to right.

---

### 5. DP State and Invariant

Use two arrays indexed by transaction count.

After processing some prefix of days, define:

```text
cash[t] = maximum profit after processing these days,
          having completed exactly t sells,
          and holding no stock

hold[t] = maximum profit after processing these days,
          having completed exactly t sells,
          and holding one stock
```

The invariant is:

```text
After each day, cash[t] and hold[t] store the best possible profit among all legal action sequences ending in that exact state.
```

This is the core of the solution.

The values are not just vague summaries. They are precise best answers for the two possible physical states after a fixed number of completed transactions.

---

### 6. What Each Transition Means

On a day with price `p`, there are four conceptual choices.

#### Keep not holding

If we were not holding before, we can skip today:

```text
cash[t] stays cash[t]
```

#### Buy today

If we were not holding before, we can buy today:

```text
hold[t] = max(hold[t], cash[t] - p)
```

Buying does not complete a transaction, so `t` stays the same.

The profit decreases by `p` because we spend money to acquire the stock.

#### Keep holding

If we were holding before, we can skip today:

```text
hold[t] stays hold[t]
```

#### Sell today

If we were holding before with `t - 1` completed transactions, selling today completes one more transaction:

```text
cash[t] = max(cash[t], hold[t - 1] + p)
```

Selling increases profit by `p` and moves us from holding to not holding.

The transaction count increases because a buy/sell pair has just been completed.

---

### 7. Initialization

Before seeing any prices:

```text
cash[0] = 0
```

With no days processed, zero profit and no stock is possible.

For all other states, use negative infinity:

```text
cash[t] = -infinity for t > 0
hold[t] = -infinity for all t
```

Why?

Before any day is processed:

```text
completed one sell without a day      -> impossible
holding a stock without buying first  -> impossible
```

Some implementations initialize `hold[t]` during the first day instead of explicitly using negative infinity. Both approaches are equivalent if the transitions preserve legality.

---

### 8. Algorithm

For each price `p`:

1. Consider every transaction count `t` from `0` to `k`.
2. Update the best holding state by either keeping the previous holding state or buying today.
3. Update the best not-holding state by either keeping the previous not-holding state or selling today.
4. After all prices, return the best `cash[t]` for `0 <= t <= k`.

Because using fewer than `k` transactions is allowed, the answer is not forced to use exactly `k` sells. It is:

```text
max(cash[0], cash[1], ..., cash[k])
```

In many standard implementations, `cash[t]` is monotonic enough that `cash[k]` also gives the answer, but returning the maximum over all completed transaction counts matches the problem statement most directly.

---

### 9. Pseudocode

One clear version uses separate arrays for the next day so the transition is easy to reason about:

```python
def maxProfit(k, prices):
    NEG_INF = float("-inf")

    cash = [NEG_INF] * (k + 1)
    hold = [NEG_INF] * (k + 1)
    cash[0] = 0

    for price in prices:
        next_cash = cash[:]
        next_hold = hold[:]

        for t in range(k + 1):
            # Buy today from a not-holding state with t completed sells.
            next_hold[t] = max(next_hold[t], cash[t] - price)

            # Sell today from a holding state with t - 1 completed sells.
            if t > 0:
                next_cash[t] = max(next_cash[t], hold[t - 1] + price)

        cash = next_cash
        hold = next_hold

    return max(cash)
```

This version is intentionally explicit. It prevents accidental same-day reuse of an updated state.

Same-day buy then sell at the same price would not improve profit, so carefully written in-place versions are also possible. But the two-array version makes the invariant easiest to see.

---

### 10. Space-Optimized In-Place Shape

A common implementation keeps only two arrays and updates them in place:

```python
def maxProfit(k, prices):
    if not prices or k == 0:
        return 0

    n = len(prices)

    # More than n // 2 completed transactions cannot be useful,
    # because each transaction needs at least one buy day and one later sell day.
    k = min(k, n // 2)

    cash = [0] * (k + 1)
    hold = [float("-inf")] * (k + 1)

    for price in prices:
        for t in range(k + 1):
            hold[t] = max(hold[t], cash[t] - price)
            if t > 0:
                cash[t] = max(cash[t], hold[t - 1] + price)

    return cash[k]
```

This compact version uses a closely related capacity interpretation:

```text
cash[t] = best completed state with permission to use at most t transactions
hold[t] = best open-position state with permission to use at most t transactions
```

That is why it can initialize every `cash[t]` to `0`: doing nothing is legal no matter how many transactions are allowed. The exact-state version is often easier for proof; the capacity-state version is often easier to code.

The line:

```python
k = min(k, n // 2)
```

is not required for correctness, but it avoids maintaining useless transaction states. With `n` days, at most `n // 2` profitable or non-overlapping buy/sell pairs can be completed.

---

### 11. Detailed Walkthrough

Use the second example:

```text
k = 2
prices = [3, 2, 6, 5, 0, 3]
```

We track:

```text
cash[0], cash[1], cash[2]
hold[0], hold[1], hold[2]
```

The most important states are:

```text
hold[0] = best balance after buying the first stock, before any sell
cash[1] = best profit after completing one transaction
hold[1] = best balance after completing one transaction, then buying again
cash[2] = best profit after completing two transactions
```

Initial state:

```text
cash = [0, 0, 0]          # in the compact form, doing nothing is allowed for every limit
hold = [-inf, -inf, -inf]
```

#### Day 0, price = 3

Buy the first stock:

```text
hold[0] = max(-inf, cash[0] - 3) = -3
```

Meaning:

```text
If we hold a stock after day 0, our balance is -3.
```

No profitable sell is possible yet.

Important states:

```text
hold[0] = -3
cash[1] = 0
hold[1] = -3
cash[2] = 0
```

In the compact `at most t transactions` form, `hold[1]` may also become `-3`, meaning: with permission to finish up to one prior transaction, the best open position is still simply buying at 3.

#### Day 1, price = 2

Buying at 2 is better than buying at 3:

```text
hold[0] = max(-3, 0 - 2) = -2
```

Now the best first buy is at price `2`.

Selling immediately after buying at 2 gives no profit, so the best completed profit remains:

```text
cash[1] = 0
cash[2] = 0
```

#### Day 2, price = 6

Sell the stock bought at 2:

```text
cash[1] = max(0, hold[0] + 6)
        = max(0, -2 + 6)
        = 4
```

Now one completed transaction can produce profit `4`:

```text
buy at 2, sell at 6
```

If we then consider a second open position, the best balance after one completed transaction and holding again is:

```text
hold[1] = max(previous hold[1], cash[1] - 6)
```

That represents finishing one transaction and buying again. Buying again at `6` is not attractive yet, but the state records the best legal possibility.

Important states after this day include:

```text
cash[1] = 4
cash[2] = 4
```

With a limit of two transactions, doing only the first good transaction is still allowed.

#### Day 3, price = 5

Selling at 5 is worse than having sold at 6:

```text
cash[1] stays 4
```

But buying the second stock at 5 after earning profit 4 gives:

```text
hold[1] = max(hold[1], cash[1] - 5)
        = max(previous, 4 - 5)
        = -1
```

This means:

```text
After buy at 2, sell at 6, then buy at 5,
our net balance is -1.
```

That is a legal second holding state, but it may improve later if the price drops.

#### Day 4, price = 0

Price `0` is an excellent second buy.

The first holding state improves too:

```text
hold[0] = max(-2, 0 - 0) = 0
```

Meaning: if we are still before any completed sell, the best open position is buying at 0.

More importantly, after the first completed transaction with profit `4`, buying at 0 gives:

```text
hold[1] = max(-1, cash[1] - 0)
        = max(-1, 4)
        = 4
```

This state means:

```text
We have completed one transaction with profit 4,
then bought another stock for 0,
so our net balance is still 4.
```

#### Day 5, price = 3

Now sell the second stock:

```text
cash[2] = max(previous cash[2], hold[1] + 3)
        = max(4, 4 + 3)
        = 7
```

This corresponds exactly to:

```text
buy at 2, sell at 6  -> +4
buy at 0, sell at 3  -> +3
```

Total:

```text
7
```

So the algorithm returns `7`.

---

### 12. Why This Handles "At Most k" Correctly

The phrase "at most" matters.

If prices are decreasing:

```text
k = 2
prices = [5, 4, 3, 2, 1]
```

The best answer is not to force a transaction. It is:

```text
0
```

The DP naturally supports this because every `cash[t]` can keep its previous value by skipping days.

So the algorithm is never forced to buy or sell. It only takes a transaction when it improves the best profit.

---

### 13. Unlimited-Transactions Shortcut

If:

```text
k >= len(prices) // 2
```

then the transaction limit is effectively not binding.

Why?

A completed transaction needs at least two distinct days:

```text
buy day < sell day
```

So with `n` days, more than `n // 2` transactions cannot be used anyway.

In that case, the problem becomes the unlimited-transactions version: collect every positive price increase between adjacent days.

For example:

```text
prices = [1, 5, 3, 6]
```

The profit is:

```text
(5 - 1) + (6 - 3) = 7
```

This shortcut can be written as:

```python
profit = 0
for i in range(1, len(prices)):
    if prices[i] > prices[i - 1]:
        profit += prices[i] - prices[i - 1]
```

However, clamping `k` to `n // 2` and running the DP is also fine, and it keeps one unified explanation.

---

### 14. Correctness Argument

We prove that the DP returns the maximum possible profit.

#### Invariant

After processing each day, for every transaction count `t`:

```text
cash[t]
```

is the maximum profit among all legal action sequences that processed those days, completed `t` sells, and end without holding stock.

And:

```text
hold[t]
```

is the maximum profit among all legal action sequences that processed those days, completed `t` sells, and end holding one stock.

#### Base Case

Before processing any prices:

```text
cash[0] = 0
```

This represents doing nothing.

All impossible states are initialized to negative infinity, or are unreachable until a legal transition creates them.

So the invariant is true before the first day.

#### Inductive Step

Assume the invariant is true before processing a day with price `p`.

For a final not-holding state with `t` completed sells, the last action on this day is one of two possibilities:

1. Do nothing while already not holding.
2. Sell today from a holding state with `t - 1` completed sells.

The transition:

```text
cash[t] = max(cash[t], hold[t - 1] + p)
```

takes the better of exactly those possibilities.

For a final holding state with `t` completed sells, the last action on this day is one of two possibilities:

1. Do nothing while already holding.
2. Buy today from a not-holding state with `t` completed sells.

The transition:

```text
hold[t] = max(hold[t], cash[t] - p)
```

takes the better of exactly those possibilities.

No other legal final state exists, because on one day the only relevant actions are skip, buy, or sell, and buying/selling is constrained by whether a stock is currently held.

Therefore the invariant remains true after processing the day.

#### Final Answer

At the end, a valid realized profit must end with no stock held. Any still-open holding state has paid for a stock that has not been sold, so it cannot be the final realized answer.

The problem allows at most `k` transactions, so the best final answer is the best not-holding state with no more than `k` completed sells.

Therefore the DP returns the maximum possible profit.

---

### 15. Complexity

Let:

```text
n = len(prices)
K = min(k, n // 2)
```

The DP processes every day and every transaction count.

Time complexity:

```text
O(n * K)
```

Space complexity:

```text
O(K)
```

because only the current `cash` and `hold` arrays are needed.

If using the unlimited-transactions shortcut when `k >= n // 2`, that shortcut runs in:

```text
O(n) time
O(1) space
```

---

### 16. Common Pitfalls

#### Counting buys instead of sells incorrectly

The clean interpretation is:

```text
t = number of completed sells
```

Buying does not increase `t`; selling does.

Mixing this up often causes off-by-one errors.

#### Returning a holding state

A state like:

```text
hold[t]
```

is not a realized profit. It includes an open stock position. The final answer must come from `cash`, not `hold`.

#### Forcing exactly k transactions

The problem says at most `k`, not exactly `k`.

If no profitable trade exists, the answer should be `0`, not a negative number.

#### Forgetting that transactions cannot overlap

This sequence is invalid:

```text
buy, buy, sell, sell
```

The holding dimension prevents that mistake because a buy transition is only allowed from a not-holding state.

#### Updating in place without understanding dependencies

In-place updates are efficient, but they can blur whether a value belongs to the previous day or the current day.

When learning or debugging, use `next_cash` and `next_hold` first. Optimize only after the transitions are clear.

#### Not handling large k

If `k` is huge relative to `n`, there are many useless transaction states.

Clamp it:

```python
k = min(k, len(prices) // 2)
```

or use the unlimited-transactions shortcut.

---

### 17. First-Principles Summary

The problem is hard because the best decision today depends on two pieces of history:

```text
how many transactions have already been completed
whether we currently hold a stock
```

Once those are included in the state, the problem becomes local.

For each price, each state has only two meaningful choices:

```text
cash[t]: stay not holding, or sell into this state
hold[t]: stay holding, or buy into this state
```

The DP works because it records the best possible profit for every complete state description. No future decision needs the exact earlier buy/sell dates; it only needs the best profit available for the current transaction count and holding status.

That is the main lesson:

> When a problem asks for the best sequence of actions under a resource limit, define the state by the resource used and the physical condition after the latest action. Then each new input item becomes a small set of legal transitions.

## Implementation

See `solutions/dynamic_programming_multidimensional/p188_best_time_to_buy_and_sell_stock_iv.py`.

## Tests

See `tests/dynamic_programming_multidimensional/test_p188_best_time_to_buy_and_sell_stock_iv.py`.

## Examples

### Example 1
- Input: `{'k': 2, 'prices': [2, 4, 1]}`
- Output: `2`

Explanation:

```text
buy at 2, sell at 4 -> profit 2
```

The last price `1` arrives after the profitable sell, so it cannot be used to increase profit.

### Example 2
- Input: `{'k': 2, 'prices': [3, 2, 6, 5, 0, 3]}`
- Output: `7`

Explanation:

```text
buy at 2, sell at 6 -> profit 4
buy at 0, sell at 3 -> profit 3
```

Total profit:

```text
4 + 3 = 7
```

## Follow-up Practice

- Trace `cash` and `hold` for `k = 1` and compare it to the simpler one-transaction stock problem.
- Trace `cash` and `hold` for `k = 2`, `prices = [3, 2, 6, 5, 0, 3]`.
- Rewrite the recurrence using a full `dp[day][transactions][holding]` table, then compress away the day dimension.
- Explain why the final answer must be a not-holding state.
