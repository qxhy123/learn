# 123. Best Time to Buy and Sell Stock III

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `prices`, where:

```text
prices[i] = the stock price on day i
```

You may complete **at most two transactions**.

One transaction means:

```text
buy once, then later sell once
```

There are two important restrictions:

1. You must buy before you sell.
2. You cannot hold multiple shares at the same time.

So this is allowed:

```text
buy -> sell -> buy -> sell
```

But this is not allowed:

```text
buy -> buy -> sell -> sell
```

The goal is to choose zero, one, or two non-overlapping buy/sell pairs that maximize total profit.

If no profitable transaction exists, the answer is `0`, because you are never forced to trade.

For example:

```text
prices = [3, 3, 5, 0, 0, 3, 1, 4]
```

One optimal plan is:

```text
buy at 3, sell at 5  -> profit 2
buy at 0, sell at 4  -> profit 4
```

Total profit:

```text
2 + 4 = 6
```

So the answer is `6`.

The real problem is:

> While scanning days in chronological order, keep the best possible profit after each stage of up to two transactions.

---

### 2. Start From the Brute Force Idea

A direct way to think about the problem is to choose the four transaction days:

```text
first_buy_day <= first_sell_day < second_buy_day <= second_sell_day
```

Then compute:

```text
prices[first_sell_day] - prices[first_buy_day]
+ prices[second_sell_day] - prices[second_buy_day]
```

We could try every possible combination of days.

Conceptually:

```python
best = 0

for b1 in range(n):
    for s1 in range(b1 + 1, n):
        first_profit = prices[s1] - prices[b1]
        best = max(best, first_profit)

        for b2 in range(s1 + 1, n):
            for s2 in range(b2 + 1, n):
                second_profit = prices[s2] - prices[b2]
                best = max(best, first_profit + second_profit)
```

This is correct in spirit, but it is far too slow.

There are too many combinations of buy and sell days. More importantly, the brute force repeats the same question many times:

```text
Before today, what was the best position I could have been in?
```

Dynamic programming works because we do not need to remember every exact buy/sell day. We only need to remember the best profit possible after each meaningful stage.

---

### 3. The Key Observation

At any point in time, a trading plan is in one of a few stages.

Because we can make at most two transactions and can hold at most one stock, the useful stages are:

```text
after first buy
after first sell
after second buy
after second sell
```

Each stage can be represented by the best profit possible after processing the days seen so far.

The surprising but crucial idea is that a "buy" state may be negative.

If you buy a stock for price `p`, your profit becomes:

```text
-p
```

because you have spent money and are holding one share.

If you later sell for price `q`, your profit becomes:

```text
-p + q
```

For two transactions, the second buy is different from the first buy. When you buy the second stock, you may already have profit from the first transaction:

```text
profit_after_first_sell - current_price
```

So the second buy is not just "lowest price so far." It is:

```text
best money after finishing transaction 1, then buying again
```

That is why this problem needs more than one scalar minimum price.

---

### 4. DP State and Invariant

We scan the price array from left to right.

After processing each day, maintain four values:

```text
first_buy   = maximum profit possible after buying the first stock
first_sell  = maximum profit possible after selling the first stock
second_buy  = maximum profit possible after buying the second stock
second_sell = maximum profit possible after selling the second stock
```

These are not necessarily actions taken today. They are the best values achievable using any days up to and including today.

The invariant after day `i` is:

```text
first_buy   is the best profit among all valid plans ending with one held stock
            after the first buy, using only days 0..i.

first_sell  is the best profit among all valid plans ending with no stock
            after at most one completed transaction, using only days 0..i.

second_buy  is the best profit among all valid plans ending with one held stock
            after one completed transaction and a second buy, using only days 0..i.

second_sell is the best profit among all valid plans ending with no stock
            after at most two completed transactions, using only days 0..i.
```

The answer will be `second_sell`.

Even if we never use two transactions, `second_sell` can still represent the best final profit because the update rules allow the second transaction to contribute nothing when it is not useful.

---

### 5. Transitions From First Principles

On each day, let:

```text
price = prices[i]
```

For each state, there are only two possibilities:

1. Do nothing today and keep the old state.
2. Perform the one action that moves into this state today.

#### Updating `first_buy`

To be after the first buy:

```text
keep previous first_buy
or buy today from starting profit 0
```

So:

```python
first_buy = max(first_buy, -price)
```

#### Updating `first_sell`

To be after the first sell:

```text
keep previous first_sell
or sell today after a previous first buy
```

So:

```python
first_sell = max(first_sell, first_buy + price)
```

#### Updating `second_buy`

To be after the second buy:

```text
keep previous second_buy
or buy today after a previous first sell
```

So:

```python
second_buy = max(second_buy, first_sell - price)
```

#### Updating `second_sell`

To be after the second sell:

```text
keep previous second_sell
or sell today after a previous second buy
```

So:

```python
second_sell = max(second_sell, second_buy + price)
```

Together:

```python
first_buy = max(first_buy, -price)
first_sell = max(first_sell, first_buy + price)
second_buy = max(second_buy, first_sell - price)
second_sell = max(second_sell, second_buy + price)
```

This is a compressed multidimensional DP.

The hidden dimensions are:

```text
day index
number of completed sells / transaction stage
holding or not holding
```

Instead of storing the whole table, we keep only the four states needed for the next day.

---

### 6. Why the Update Order Works

The update order is:

```text
first_buy -> first_sell -> second_buy -> second_sell
```

This follows the chronological transaction stages.

Using the newly updated value from the same day can appear suspicious at first. For example, `first_sell` may use `first_buy` that was updated with today's price.

That is safe because buying and selling on the same day creates zero profit:

```text
-price + price = 0
```

It does not create an illegal advantage. It simply allows the DP to model "start or skip a transaction here" without special cases.

The same reasoning applies to selling the first transaction and buying the second transaction on the same day. That sequence changes no profit on that day and is equivalent to ending one transaction and being ready for the next.

---

### 7. Algorithm

1. If `prices` is empty, return `0`.

2. Initialize the four states:

```python
first_buy = -prices[0]
first_sell = 0
second_buy = -prices[0]
second_sell = 0
```

Why these values?

- `first_buy = -prices[0]`: buying on day `0` costs `prices[0]`.
- `first_sell = 0`: no completed profitable transaction yet.
- `second_buy = -prices[0]`: this can be interpreted as making a zero-profit first transaction and then buying on day `0`; it is a convenient valid initialization.
- `second_sell = 0`: no completed profitable transaction yet.

3. For each price, update the four states in order.

4. Return `second_sell`.

---

### 8. Python Code

```python
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        first_buy = -prices[0]
        first_sell = 0
        second_buy = -prices[0]
        second_sell = 0

        for price in prices:
            first_buy = max(first_buy, -price)
            first_sell = max(first_sell, first_buy + price)
            second_buy = max(second_buy, first_sell - price)
            second_sell = max(second_sell, second_buy + price)

        return second_sell
```

Equivalent pseudocode:

```text
if prices is empty:
    return 0

first_buy = -prices[0]
first_sell = 0
second_buy = -prices[0]
second_sell = 0

for price in prices:
    first_buy = max(first_buy, -price)
    first_sell = max(first_sell, first_buy + price)
    second_buy = max(second_buy, first_sell - price)
    second_sell = max(second_sell, second_buy + price)

return second_sell
```

---

### 9. Detailed Example Walkthrough

Use the main example:

```text
prices = [3, 3, 5, 0, 0, 3, 1, 4]
```

The four states mean:

```text
first_buy   = best after first buy
first_sell  = best after first sell
second_buy  = best after second buy
second_sell = best after second sell
```

Start with day `0`, price `3`:

```text
first_buy = -3
first_sell = 0
second_buy = -3
second_sell = 0
```

Now scan each price.

#### Day 0, price = 3

```text
first_buy   = max(-3, -3)     = -3
first_sell  = max(0, -3 + 3)  = 0
second_buy  = max(-3, 0 - 3)  = -3
second_sell = max(0, -3 + 3)  = 0
```

No profit yet.

#### Day 1, price = 3

The price is unchanged:

```text
first_buy   = -3
first_sell  = 0
second_buy  = -3
second_sell = 0
```

Still no profit.

#### Day 2, price = 5

```text
first_buy   = max(-3, -5)     = -3
first_sell  = max(0, -3 + 5)  = 2
second_buy  = max(-3, 2 - 5)  = -3
second_sell = max(0, -3 + 5)  = 2
```

The best one-transaction profit is now:

```text
buy at 3, sell at 5 -> profit 2
```

`second_sell` is also `2` because using only one transaction is allowed.

#### Day 3, price = 0

```text
first_buy   = max(-3, 0)      = 0
first_sell  = max(2, 0 + 0)   = 2
second_buy  = max(-3, 2 - 0)  = 2
second_sell = max(2, 2 + 0)   = 2
```

Two important things happen:

```text
first_buy = 0
```

means the best first buy is now buying at price `0`.

```text
second_buy = 2
```

means we can keep the earlier profit `2` from selling at `5`, then buy again at price `0`. After that second buy, we are still effectively up `2`.

#### Day 4, price = 0

The states stay the same:

```text
first_buy   = 0
first_sell  = 2
second_buy  = 2
second_sell = 2
```

Another day with price `0` gives the same opportunities.

#### Day 5, price = 3

```text
first_buy   = max(0, -3)      = 0
first_sell  = max(2, 0 + 3)   = 3
second_buy  = max(2, 3 - 3)   = 2
second_sell = max(2, 2 + 3)   = 5
```

Now:

```text
second_sell = 5
```

One plan achieving this is:

```text
buy at 3, sell at 5 -> profit 2
buy at 0, sell at 3 -> profit 3
```

Total:

```text
5
```

#### Day 6, price = 1

```text
first_buy   = max(0, -1)      = 0
first_sell  = max(3, 0 + 1)   = 3
second_buy  = max(2, 3 - 1)   = 2
second_sell = max(5, 2 + 1)   = 5
```

The best total profit remains `5`.

Buying the second stock at `1` after a first-sell profit of `3` gives:

```text
3 - 1 = 2
```

which ties the existing `second_buy`.

#### Day 7, price = 4

```text
first_buy   = max(0, -4)      = 0
first_sell  = max(3, 0 + 4)   = 4
second_buy  = max(2, 4 - 4)   = 2
second_sell = max(5, 2 + 4)   = 6
```

Final answer:

```text
6
```

One optimal plan is:

```text
buy at 3, sell at 5 -> profit 2
buy at 0, sell at 4 -> profit 4
```

Total:

```text
6
```

---

### 10. Correctness Argument

We prove that the algorithm returns the maximum profit obtainable with at most two transactions.

#### Invariant

After processing each day, the four variables have these meanings:

```text
first_buy   = best profit after a first buy and currently holding one stock
first_sell  = best profit after at most one completed transaction and holding no stock
second_buy  = best profit after one completed transaction, a second buy, and currently holding one stock
second_sell = best profit after at most two completed transactions and holding no stock
```

#### Initialization

Before any profitable sale is possible:

```text
first_sell = 0
second_sell = 0
```

This represents doing nothing, which is always allowed.

On the first day:

```text
first_buy = -prices[0]
second_buy = -prices[0]
```

This represents buying at the first price, either as the first buy or as a second-buy state after a zero-profit first transaction. Therefore, the invariant is true initially.

#### Maintenance

Assume the invariant is true before processing a price `price`.

For `first_buy`, any best plan after the first buy either:

```text
already existed before today
or buys today from profit 0
```

The update `max(first_buy, -price)` chooses the better of exactly those possibilities.

For `first_sell`, any best plan after the first sell either:

```text
already existed before today
or sells today from a valid first_buy state
```

The update `max(first_sell, first_buy + price)` chooses the better of exactly those possibilities.

For `second_buy`, any best plan after the second buy either:

```text
already existed before today
or buys today from a valid first_sell state
```

The update `max(second_buy, first_sell - price)` chooses the better of exactly those possibilities.

For `second_sell`, any best plan after the second sell either:

```text
already existed before today
or sells today from a valid second_buy state
```

The update `max(second_sell, second_buy + price)` chooses the better of exactly those possibilities.

Thus each update preserves the invariant.

#### Termination

After all days are processed, `second_sell` is the best profit among all valid plans that finish with no stock after at most two transactions.

An optimal answer must finish with no stock, because holding an unsold stock cannot increase realized profit. Therefore, `second_sell` is exactly the required maximum profit.

---

### 11. Complexity

Let `n = len(prices)`.

Each day performs a constant number of updates.

```text
Time complexity:  O(n)
Space complexity: O(1)
```

The full DP table would have dimensions for day, transaction stage, and holding status, but the transition only needs the previous best values. That is why the solution can be compressed to four variables.

---

### 12. Common Pitfalls

#### Pitfall 1: Treating the answer as exactly two transactions

The problem says **at most** two transactions.

If prices only increase:

```text
prices = [1, 2, 3, 4, 5]
```

The best answer is one transaction:

```text
buy at 1, sell at 5 -> profit 4
```

Forcing two transactions is unnecessary.

#### Pitfall 2: Forgetting that you cannot hold two stocks

You cannot do:

```text
buy first stock
buy second stock
sell later
```

The state order prevents this by requiring:

```text
first_sell before second_buy
```

#### Pitfall 3: Thinking `second_buy` must be negative

`second_buy` can be positive.

For example, after earning profit `2` from the first transaction, buying at price `0` gives:

```text
second_buy = 2 - 0 = 2
```

This means you are holding a stock while still having net profit from the earlier transaction.

#### Pitfall 4: Updating only a minimum price

For the one-transaction stock problem, tracking the minimum price is enough.

For two transactions, the second buy depends on previous profit:

```text
first_sell - price
```

So a simple global minimum price loses information.

#### Pitfall 5: Returning a holding state

Do not return `first_buy` or `second_buy`.

A holding state includes the cost of a stock that has not been sold. The final answer must be realized profit, so it must be a non-holding state.

---

### 13. First-Principles Summary

The problem is hard only if we try to remember exact buy and sell days.

The first-principles move is to ask:

> What information about the past is sufficient to make the next decision?

For this problem, the sufficient information is the best profit after each transaction stage:

```text
first buy
first sell
second buy
second sell
```

Each new price offers one local decision for each stage:

```text
keep the old state
or perform the action that enters this state today
```

That gives the four-state recurrence:

```text
first_buy   = max(first_buy, -price)
first_sell  = max(first_sell, first_buy + price)
second_buy  = max(second_buy, first_sell - price)
second_sell = max(second_sell, second_buy + price)
```

After all prices are processed, `second_sell` is the best realized profit with at most two transactions.

## Implementation
See `solutions/dynamic_programming_multidimensional/p123_best_time_to_buy_and_sell_stock_iii.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p123_best_time_to_buy_and_sell_stock_iii.py`.

## Examples

### Example 1
- Input: `{'prices': [3, 3, 5, 0, 0, 3, 1, 4]}`
- Output: `6`

### Example 2
- Input: `{'prices': [1, 2, 3, 4, 5]}`
- Output: `4`

### Example 3
- Input: `{'prices': [7, 6, 4, 3, 1]}`
- Output: `0`

## Follow-up Practice
- Trace the four state variables for `prices = [1, 2, 4, 2, 5, 7, 2, 4, 9, 0]`.
- Rewrite the four-variable solution as a full `day x transaction x holding` DP table.
- Explain why `second_buy` can be positive.
- Compare this problem with the one-transaction stock problem and identify what extra state is needed here.
