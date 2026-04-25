# 122. Best Time to Buy and Sell Stock II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, peak-valley

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array:

```text
prices[i] = stock price on day i
```

You may make as many transactions as you want, where one transaction means:

```text
buy once, then sell later
```

But there is one important restriction:

```text
You may hold at most one share at a time.
```

So you cannot buy on day `1`, buy again on day `2`, and then sell both later. Before buying again, the previous share must already have been sold.

The goal is:

```text
maximize total profit
```

For example:

```text
prices = [7, 1, 5, 3, 6, 4]
```

One good strategy is:

```text
buy at 1, sell at 5  -> profit 4
buy at 3, sell at 6  -> profit 3
total profit         -> 7
```

The problem is not asking for the single best buy/sell pair. That is LeetCode 121. Here, multiple non-overlapping profitable trades are allowed.

So the real question is:

> Given a sequence of prices, what is the maximum profit obtainable by repeatedly owning either zero shares or one share?

---

### 2. Start From the Brute Force Idea

A direct way to think about the problem is decision-based:

On each day, if you are not holding a share, you may:

```text
do nothing
buy
```

If you are holding a share, you may:

```text
do nothing
sell
```

This creates a tree of possibilities.

For each day, the future depends on whether we currently hold a stock. A recursive brute force version would explore many combinations:

```python
def search(day, holding):
    if day == len(prices):
        return 0 if not holding else -infinity

    if holding:
        return max(
            search(day + 1, True),                 # keep holding
            prices[day] + search(day + 1, False),  # sell today
        )

    return max(
        search(day + 1, False),                    # stay out of market
        -prices[day] + search(day + 1, True),      # buy today
    )
```

This is conceptually correct because it considers every legal sequence of buys and sells.

But it is much too slow: each day can branch, so the number of paths grows exponentially.

We can improve this with dynamic programming by storing two states:

```text
cash = best profit after this day when holding no stock
hold = best profit after this day when holding one stock
```

That gives an `O(n)` solution.

But this particular problem has an even simpler structure. Because transactions are unlimited, we do not need to remember which transaction number we are on. We only need to understand which price changes are worth taking.

---

### 3. Look At Profit As Day-To-Day Movement

Suppose prices rise across several days:

```text
prices = [1, 2, 3, 4, 5]
```

One obvious strategy is:

```text
buy at 1, sell at 5 -> profit 4
```

But the same profit can also be written as daily increases:

```text
(2 - 1) + (3 - 2) + (4 - 3) + (5 - 4)
= 1 + 1 + 1 + 1
= 4
```

This is not a coincidence. For any buy day `b` and sell day `s`:

```text
prices[s] - prices[b]
```

can be expanded into:

```text
(prices[b + 1] - prices[b])
+ (prices[b + 2] - prices[b + 1])
+ ...
+ (prices[s] - prices[s - 1])
```

The intermediate prices cancel out.

That means a long trade is just a sum of adjacent price changes.

So if a long trade is profitable, its profit comes from the positive movement between days. The question becomes:

> Which adjacent price movements should be included?

---

### 4. The Key Observation

If tomorrow's price is higher than today's price:

```text
prices[i] > prices[i - 1]
```

then the increase:

```text
prices[i] - prices[i - 1]
```

is profit we can safely capture.

Why is it safe?

Because we are allowed to complete one transaction and then start another. Capturing an increase from day `i - 1` to day `i` can be interpreted as:

```text
buy on day i - 1
sell on day i
```

If prices keep rising, doing this every day gives the same profit as buying once at the valley and selling once at the peak.

For example:

```text
[1, 2, 3, 4]
```

Daily trades:

```text
buy 1, sell 2 -> 1
buy 2, sell 3 -> 1
buy 3, sell 4 -> 1
total         -> 3
```

One long trade:

```text
buy 1, sell 4 -> 3
```

Both are legal in terms of profit. Even though a real execution would not need to sell and rebuy every day, the profit calculation is equivalent.

If tomorrow's price is lower or equal:

```text
prices[i] <= prices[i - 1]
```

then there is no positive profit in holding from yesterday to today. Taking that edge would reduce profit or change nothing, so an optimal strategy can skip it.

This gives the greedy rule:

```text
Add every positive adjacent difference.
Ignore every zero or negative adjacent difference.
```

---

### 5. Peak-Valley View

Another way to see the same idea is with valleys and peaks.

In:

```text
[7, 1, 5, 3, 6, 4]
```

the profitable rising segments are:

```text
1 -> 5
3 -> 6
```

So the best strategy is:

```text
buy at each valley
sell at the next peak
```

The profit is:

```text
(5 - 1) + (6 - 3) = 4 + 3 = 7
```

The adjacent-difference algorithm computes the same thing:

```text
7 -> 1  difference -6  ignore
1 -> 5  difference +4  add
5 -> 3  difference -2  ignore
3 -> 6  difference +3  add
6 -> 4  difference -2  ignore

total = 4 + 3 = 7
```

So the algorithm does not explicitly search for valleys and peaks. It captures them implicitly by adding every upward step.

---

### 6. The State And Invariant

The algorithm keeps one piece of state:

```text
profit = total profit from all positive adjacent increases seen so far
```

After processing the transition from day `i - 1` to day `i`, maintain this invariant:

```text
profit equals the maximum profit obtainable using only days 0 through i,
assuming all completed profit comes from the positive price movements already seen.
```

More concretely:

```text
profit = sum(max(0, prices[j] - prices[j - 1])) for j from 1 through i
```

This invariant is enough because unlimited transactions remove the need to preserve a limited transaction budget. Every positive edge can be included independently, and every non-positive edge can be excluded independently.

At each new day, exactly one new adjacent movement becomes known:

```text
delta = prices[i] - prices[i - 1]
```

There are only two cases:

```text
delta > 0   -> add delta
delta <= 0  -> add nothing
```

After that update, the invariant is true for the larger prefix.

---

### 7. Detailed Algorithm

1. If the list has fewer than two prices, return `0`.

   With zero or one day, there is no later day to sell after buying, so no profit is possible.

2. Initialize:

```text
profit = 0
```

3. Scan from the second price to the end.

4. For each day `i`, compute:

```text
increase = prices[i] - prices[i - 1]
```

5. If `increase` is positive, add it:

```text
profit += increase
```

6. If `increase` is zero or negative, ignore it.

7. Return `profit`.

In Python-like pseudocode:

```python
def maxProfit(prices):
    profit = 0

    for i in range(1, len(prices)):
        increase = prices[i] - prices[i - 1]
        if increase > 0:
            profit += increase

    return profit
```

The same idea can be written more compactly:

```python
def maxProfit(prices):
    profit = 0

    for previous, current in zip(prices, prices[1:]):
        profit += max(0, current - previous)

    return profit
```

---

### 8. Example Walkthrough: `[7, 1, 5, 3, 6, 4]`

Start:

```text
profit = 0
```

Compare day `0` to day `1`:

```text
7 -> 1
increase = 1 - 7 = -6
```

This is a drop. Holding stock across this edge would lose value.

```text
profit = 0
```

Compare day `1` to day `2`:

```text
1 -> 5
increase = 5 - 1 = 4
```

This is a rise, so capture it.

```text
profit = 0 + 4 = 4
```

Compare day `2` to day `3`:

```text
5 -> 3
increase = 3 - 5 = -2
```

This is a drop, so skip it.

```text
profit = 4
```

Compare day `3` to day `4`:

```text
3 -> 6
increase = 6 - 3 = 3
```

This is a rise, so capture it.

```text
profit = 4 + 3 = 7
```

Compare day `4` to day `5`:

```text
6 -> 4
increase = 4 - 6 = -2
```

This is a drop, so skip it.

```text
profit = 7
```

Return:

```text
7
```

This corresponds to:

```text
buy at 1, sell at 5
buy at 3, sell at 6
```

---

### 9. Why Greedy Is Correct

We need to prove that adding every positive adjacent increase gives the maximum possible profit.

Any legal strategy is a set of non-overlapping buy/sell intervals:

```text
buy at b1, sell at s1
buy at b2, sell at s2
...
```

where:

```text
b1 < s1 < b2 < s2 < ...
```

The profit of one transaction from day `b` to day `s` is:

```text
prices[s] - prices[b]
```

That can be expanded into adjacent differences:

```text
(prices[b + 1] - prices[b])
+ (prices[b + 2] - prices[b + 1])
+ ...
+ (prices[s] - prices[s - 1])
```

Inside that sum, any negative adjacent difference makes the transaction worse. Because transactions are unlimited, an optimal strategy never needs to carry stock across a price drop:

```text
sell before the drop
buy again after the drop
```

This is legal because we still hold at most one share at a time.

Therefore, every optimal strategy can be transformed into another strategy that keeps all positive adjacent differences it uses and removes all negative adjacent differences, without decreasing profit.

Also, every positive adjacent difference can be realized legally:

```text
buy on day i - 1
sell on day i
```

or as part of a longer rising segment. Combining adjacent rising days does not change the total profit because the differences telescope.

So:

1. No strategy can earn more than the sum of all positive adjacent differences, because every transaction decomposes into adjacent differences and negative ones cannot help.
2. The greedy algorithm earns exactly the sum of all positive adjacent differences.

Therefore, the greedy algorithm returns the maximum possible profit.

---

### 10. Complexity

Let:

```text
n = len(prices)
```

The algorithm compares each adjacent pair once.

Time complexity:

```text
O(n)
```

It uses only one running total.

Space complexity:

```text
O(1)
```

---

### 11. Common Pitfalls

- Treating this like the single-transaction version.

  For `[7, 1, 5, 3, 6, 4]`, the best single transaction gives `5`, from buying at `1` and selling at `6`. But multiple transactions give `7`.

- Trying to find one global minimum and one global maximum.

  That misses separate profitable segments.

- Thinking daily positive differences violate the "hold at most one share" rule.

  They do not. A rising segment can be viewed either as one long transaction or as equivalent adjacent gains. The profit is the same.

- Adding negative differences.

  A price drop should never be included because selling before the drop and buying after it is at least as good.

- Forgetting short inputs.

  If `prices` is empty or has one element, the loop naturally returns `0`.

- Overcomplicating the solution with a transaction count.

  There is no limit on the number of transactions, so no transaction-index state is needed.

---

### 12. First-Principles Summary

The price array is a sequence of daily movements.

Profit does not come from the absolute prices by themselves. Profit comes from moving from a lower price to a higher later price.

Because transactions are unlimited and only one share may be held, every profitable rising segment can be split into adjacent rising edges without changing the profit.

So the maximum profit is exactly:

```text
sum of every positive prices[i] - prices[i - 1]
```

The algorithm is a one-pass scan that maintains:

```text
profit = all upward movement seen so far
```

Whenever the next day is higher than the previous day, add that increase. Otherwise, ignore it.

That is why the solution is short: the hard part is not implementation, but recognizing that unlimited transactions turn the problem into collecting all upward movement in the price sequence.

## Implementation

See `solutions/array_string/p122_best_time_to_buy_and_sell_stock_ii.py`.

## Tests

See `tests/array_string/test_p122_best_time_to_buy_and_sell_stock_ii.py`.

## Examples

### Example 1
- Input: `{'prices': [7, 1, 5, 3, 6, 4]}`
- Output: `7`

### Example 2
- Input: `{'prices': [1, 2, 3, 4, 5]}`
- Output: `4`

### Example 3
- Input: `{'prices': [7, 6, 4, 3, 1]}`
- Output: `0`

## Follow-up Practice

- Trace `profit` after each adjacent pair.
- Compare the greedy result with the two-state dynamic programming result.
- Explain why a falling edge should be skipped rather than held through.
