# 121. Best Time to Buy and Sell Stock

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, running-minimum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `prices` where `prices[i]` is the stock price on day `i`.

You may make **exactly one buy** and **exactly one sell**, with the restriction that:

- you must buy before you sell
- you can choose not to trade if every trade loses money

So the real question is:

> Among all pairs of days `buy < sell`, what is the largest value of `prices[sell] - prices[buy]`?

If every such value is negative, the answer should be `0`, because doing nothing is better than losing money.

This is not a simulation problem about many transactions. It is a much narrower question:

> For each possible selling day, what is the best earlier day to have bought?

Once you see the problem that way, the solution becomes much smaller.

---

### 2. Start From the Brute-Force Baseline

The most direct approach is to try every legal buy day and every legal sell day after it.

Conceptually:

```python
best = 0

for buy in range(len(prices)):
    for sell in range(buy + 1, len(prices)):
        profit = prices[sell] - prices[buy]
        best = max(best, profit)

return best
```

Why this works:

- every valid transaction is a pair `(buy, sell)` with `buy < sell`
- the algorithm checks all such pairs
- `best` keeps the maximum profit seen so far

Why this is too slow:

- there are about `n * (n - 1) / 2` buy/sell pairs
- that is `O(n^2)` time

For a small array, this is fine. For large inputs, it is unnecessary because most of that work repeats the same idea:

When we consider selling on day `j`, we do not really care about **all** earlier prices individually. We only care about the **smallest** earlier price.

---

### 3. Key Observation

Fix a selling day.

Suppose today the price is `6`, and the earlier prices were:

```text
7, 1, 5, 3
```

If you sell today at `6`, your profit depends on which earlier day you bought:

```text
buy at 7 -> profit = -1
buy at 1 -> profit = 5
buy at 5 -> profit = 1
buy at 3 -> profit = 3
```

The best choice is obviously the cheapest earlier price, which is `1`.

That means:

> For any fixed selling day, the only earlier information that matters is the minimum price seen before that day.

So instead of remembering every previous day, we can compress the entire prefix into one number:

```text
min_price_so_far
```

Then the best profit from selling today is simply:

```text
current_price - min_price_so_far
```

This turns an `O(n^2)` search over pairs into a one-pass scan.

---

### 4. The State We Need to Maintain

As we scan from left to right, maintain two pieces of state:

```text
min_price   = the cheapest price seen in all previous days
best_profit = the best valid profit seen so far
```

Those two values are enough because:

- `min_price` tells us the best possible buy for the current sell day
- `best_profit` remembers the best completed transaction across all days processed so far

The important invariant is:

> Before processing the current day as a selling day, `min_price` equals the minimum price among all earlier days, and `best_profit` equals the maximum profit obtainable using only days already processed.

Once that invariant is true, the current day is easy:

1. Treat today as the sell day.
2. Compute profit if we bought at the cheapest earlier day.
3. Update `best_profit`.
4. Then update `min_price` so future days may buy at today's price if it is cheaper.

That order is conceptually clean because it keeps "buy before sell" explicit.

---

### 5. Detailed Algorithm

If the array has fewer than two prices, no transaction is possible, so the answer is `0`.

Otherwise:

1. Set:

```text
min_price = prices[0]
best_profit = 0
```

2. Scan days `1` through `n - 1`.

3. On each day with price `price`:

```text
candidate_profit = price - min_price
best_profit = max(best_profit, candidate_profit)
min_price = min(min_price, price)
```

4. Return `best_profit`.

Why this is enough:

- `candidate_profit` is the best profit for selling today, because `min_price` is the cheapest earlier buy
- `best_profit` compares today's best sale against every earlier best sale
- updating `min_price` prepares the state for later days

---

### 6. Pseudocode

```python
def maxProfit(prices):
    if len(prices) < 2:
        return 0

    min_price = prices[0]
    best_profit = 0

    for price in prices[1:]:
        best_profit = max(best_profit, price - min_price)
        min_price = min(min_price, price)

    return best_profit
```

Equivalent formulation:

- some solutions update `min_price` first
- some only compute profit when `price > min_price`

These are all the same idea: keep a running minimum and compare every later price against it.

---

### 7. Walk Through Example 1 Carefully

Input:

```text
prices = [7, 1, 5, 3, 6, 4]
```

Initialize:

```text
min_price = 7
best_profit = 0
```

Now process each later day.

#### Day 1, price = 1

If we sell today:

```text
candidate_profit = 1 - 7 = -6
```

That is worse than doing nothing, so:

```text
best_profit = max(0, -6) = 0
```

Update the cheapest buy seen so far:

```text
min_price = min(7, 1) = 1
```

State now:

```text
min_price = 1
best_profit = 0
```

#### Day 2, price = 5

If we sell today and buy at the cheapest earlier day:

```text
candidate_profit = 5 - 1 = 4
```

Update:

```text
best_profit = max(0, 4) = 4
min_price = min(1, 5) = 1
```

State now:

```text
min_price = 1
best_profit = 4
```

#### Day 3, price = 3

```text
candidate_profit = 3 - 1 = 2
best_profit = max(4, 2) = 4
min_price = min(1, 3) = 1
```

State now:

```text
min_price = 1
best_profit = 4
```

#### Day 4, price = 6

```text
candidate_profit = 6 - 1 = 5
best_profit = max(4, 5) = 5
min_price = min(1, 6) = 1
```

State now:

```text
min_price = 1
best_profit = 5
```

#### Day 5, price = 4

```text
candidate_profit = 4 - 1 = 3
best_profit = max(5, 3) = 5
min_price = min(1, 4) = 1
```

Final answer:

```text
5
```

This corresponds to:

- buy at price `1`
- sell at price `6`

---

### 8. Walk Through Example 2

Input:

```text
prices = [7, 6, 4, 3, 1]
```

Every later price is lower than every earlier candidate buy.

Track the state:

```text
start: min_price = 7, best_profit = 0
price 6: candidate = -1, best_profit = 0, min_price = 6
price 4: candidate = -2, best_profit = 0, min_price = 4
price 3: candidate = -1, best_profit = 0, min_price = 3
price 1: candidate = -2, best_profit = 0, min_price = 1
```

No profitable transaction ever appears, so the algorithm correctly returns:

```text
0
```

That matches the problem statement: if profit is impossible, do not trade.

---

### 9. Why the Algorithm Is Correct

We can justify the algorithm directly from the invariant.

After processing day `i`:

- `min_price` is the minimum of `prices[0..i]`
- `best_profit` is the maximum profit obtainable by one valid transaction whose sell day is at most `i`

Why this stays true:

1. Before we process day `i`, `min_price` is the cheapest earlier price.
2. Therefore `prices[i] - min_price` is the best profit of any transaction that sells exactly on day `i`.
3. Taking:

```text
max(previous best_profit, prices[i] - min_price)
```

means `best_profit` now covers:

- all best transactions ending before day `i`
- the best transaction ending on day `i`

So it covers all valid transactions up to day `i`.

Then we update:

```text
min_price = min(min_price, prices[i])
```

which makes `min_price` correct for the next iteration.

Because every valid transaction has some sell day, and the algorithm evaluates the best possible buy for every sell day exactly once, the final `best_profit` is the maximum achievable profit.

---

### 10. Complexity

- Time: `O(n)`
- Space: `O(1)`

We scan once, and each step does constant work.

Compared with brute force:

- brute force: `O(n^2)` time
- running minimum: `O(n)` time

The speedup comes from compressing all earlier buy candidates into a single number.

---

### 11. Common Pitfalls

#### Forgetting the order constraint

You must buy before you sell.

This is why we scan left to right and keep only earlier prices in `min_price`.

#### Returning a negative profit

The answer should never be negative.

If every trade loses money, return:

```text
0
```

That is why `best_profit` starts at `0`.

#### Thinking the problem asks for multiple transactions

It does not.

You only get one buy and one sell. Problems like Stock II, III, and IV are different because they allow more transactions or impose different limits.

#### Storing too much information

You do not need:

- all previous prices
- a stack
- dynamic programming tables
- nested loops

For each selling day, only the cheapest earlier buy matters.

#### Mishandling very short input

If there are fewer than two days, no legal transaction exists, so the answer is `0`.

---

### 12. First-Principles Summary

The problem looks like it asks for the best pair of days, but the deeper structure is simpler:

- pick a sell day
- the best buy day is just the cheapest earlier day

So the entire prefix of the array can be summarized by one value:

```text
the minimum price seen so far
```

That gives a clean one-pass algorithm:

1. remember the cheapest price seen so far
2. treat each new day as a possible sell day
3. compute the profit from buying at that cheapest earlier day
4. keep the best profit ever seen

This is why the solution works:

> each day only asks one question: "If I sell today, what is the best earlier buy I could have made?"

Once that question is answered in constant time, the whole problem becomes linear.

## Implementation

See `solutions/array_string/p121_best_time_to_buy_and_sell_stock.py`.

## Tests

See `tests/array_string/test_p121_best_time_to_buy_and_sell_stock.py`.

## Examples

### Example 1
- Input: `{'prices': [7, 1, 5, 3, 6, 4]}`
- Output: `5`

### Example 2
- Input: `{'prices': [7, 6, 4, 3, 1]}`
- Output: `0`

## Follow-up Practice
- Trace `min_price` and `best_profit` after each day.
- Write the `O(n^2)` brute-force version first, then compare it to the one-pass version.
- Explain in one sentence why only the minimum earlier price matters for a fixed sell day.
