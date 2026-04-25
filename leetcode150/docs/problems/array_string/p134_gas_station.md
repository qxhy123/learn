# 134. Gas Station

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/gas-station/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, prefix-sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two arrays of the same length:

```text
gas[i]  = how much fuel you can collect at station i
cost[i] = how much fuel it takes to drive from station i to station i + 1
```

The stations form a circle, so after the last station you must return to station `0`.

You start with:

```text
0 fuel in the tank
```

The question is:

> Is there an index `start` such that if you begin there, refuel at each station, and drive one full loop, your tank never becomes negative?

If such a station exists, return its index. Otherwise return `-1`.

This is not asking for:

- the station with the most gas
- the station with the largest immediate profit
- the first place where `gas[i] >= cost[i]`

It is asking for a starting point whose entire circular journey is feasible.

---

### 2. Start From the Brute-Force Baseline

The most direct idea is:

1. Try every station as the starting point.
2. Simulate the trip around the full circle.
3. Keep a running `tank`.
4. If `tank` ever becomes negative, that start fails.
5. If you complete the loop, return that start.

Pseudocode:

```python
for start in range(n):
    tank = 0
    ok = True

    for step in range(n):
        i = (start + step) % n
        tank += gas[i]
        tank -= cost[i]
        if tank < 0:
            ok = False
            break

    if ok:
        return start

return -1
```

This is correct, but it can take `O(n^2)` time because each of the `n` starting points may simulate almost the whole circuit.

The real challenge is to avoid re-simulating large stretches of the circle.

---

### 3. Rewrite the Problem as Net Gain Per Station

At station `i`, what matters is not `gas[i]` and `cost[i]` separately, but their difference:

```text
delta[i] = gas[i] - cost[i]
```

If:

```text
delta[i] > 0
```

then station `i` gives you surplus fuel.

If:

```text
delta[i] < 0
```

then station `i` causes a deficit.

Now the trip becomes:

> Find a place to start so that the running sum of `delta` around the circle never goes below `0`.

This reframing exposes two crucial facts.

#### Fact 1: If the total sum is negative, no answer exists

If:

```text
sum(gas) < sum(cost)
```

then the entire circle does not contain enough fuel to pay for one full loop.

No clever starting point can fix a global shortage.

So:

```text
if sum(delta) < 0:
    return -1
```

#### Fact 2: A locally failed segment can be discarded all at once

This is the greedy heart of the problem.

---

### 4. The Key Observation: One Failed Segment Eliminates Many Starts

Suppose you choose a candidate start `s` and move forward.

Let the running balance from `s` to the current station be:

```text
current_balance = delta[s] + delta[s + 1] + ... + delta[i]
```

Now imagine that for the first time this becomes negative at station `i`.

That means:

```text
delta[s] + delta[s + 1] + ... + delta[i] < 0
```

Why does that matter?

Because then **none** of the stations from `s` through `i` can be a valid starting point.

Why?

Take any station `k` in the range:

```text
s <= k <= i
```

When we first failed at `i`, every earlier partial sum from `s` up to `k - 1` was nonnegative. Otherwise we would have failed before `i`.

So:

```text
delta[s] + ... + delta[k - 1] >= 0
```

Then the sum from `k` to `i` is:

```text
(delta[s] + ... + delta[i]) - (delta[s] + ... + delta[k - 1])
```

That is:

```text
negative - nonnegative = negative
```

So starting at `k` would also run out of fuel by the time you reach `i`.

This gives a powerful rule:

> If starting from `s` first fails at `i`, then every station from `s` to `i` is impossible. The next possible candidate is `i + 1`.

That is why the problem can be solved in one pass.

---

### 5. Invariant and State

We scan once from left to right and maintain three values:

```text
total_balance   = total net gas over all stations seen so far
current_balance = net gas from the current candidate start up to the current index
start           = current candidate starting station
```

The invariant is:

> `start` is the first station after the most recent failed segment, and `current_balance` is the net fuel balance from `start` to the current index.

Whenever:

```text
current_balance < 0
```

we have proved that the entire segment from `start` through the current index cannot contain a valid answer. So we:

```text
start = current_index + 1
current_balance = 0
```

We do **not** throw away `total_balance`, because it answers the global feasibility question.

---

### 6. The Algorithm

1. Initialize:

```text
start = 0
current_balance = 0
total_balance = 0
```

2. For each station `i`:

```text
gain = gas[i] - cost[i]
total_balance += gain
current_balance += gain
```

3. If `current_balance < 0`, then:

- the current candidate `start` fails
- every station from `start` through `i` also fails
- the next candidate must be `i + 1`

So reset:

```text
start = i + 1
current_balance = 0
```

4. After the scan:

- if `total_balance < 0`, return `-1`
- otherwise return `start`

That is the whole solution.

---

### 7. Detailed Walkthrough of Example 1

Input:

```text
gas  = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
```

Compute net gain at each station:

```text
delta = [-2, -2, -2, +3, +3]
```

Now scan left to right.

#### Start

```text
start = 0
current_balance = 0
total_balance = 0
```

#### Station 0

```text
gain = 1 - 3 = -2
total_balance = -2
current_balance = -2
```

`current_balance` is negative, so starting at `0` fails immediately.

Discard this segment and move the candidate start:

```text
start = 1
current_balance = 0
```

#### Station 1

```text
gain = 2 - 4 = -2
total_balance = -4
current_balance = -2
```

Again negative, so station `1` also cannot work:

```text
start = 2
current_balance = 0
```

#### Station 2

```text
gain = 3 - 5 = -2
total_balance = -6
current_balance = -2
```

Negative again:

```text
start = 3
current_balance = 0
```

So far we have eliminated stations `0`, `1`, and `2`.

#### Station 3

```text
gain = 4 - 1 = +3
total_balance = -3
current_balance = +3
```

This candidate is still alive.

#### Station 4

```text
gain = 5 - 2 = +3
total_balance = 0
current_balance = +6
```

The scan ends with:

```text
total_balance = 0
start = 3
```

Because `total_balance >= 0`, a solution exists, and the algorithm returns:

```text
3
```

Why does starting at `3` really work?

Simulate the full circle:

```text
start at 3: tank = 0
station 3: tank += 4, tank -= 1 => 3
station 4: tank += 5, tank -= 2 => 6
station 0: tank += 1, tank -= 3 => 4
station 1: tank += 2, tank -= 4 => 2
station 2: tank += 3, tank -= 5 => 0
```

The tank never becomes negative, so `3` is valid.

---

### 8. Why Example 2 Has No Solution

Input:

```text
gas  = [2, 3, 4]
cost = [3, 4, 3]
```

Net gains:

```text
delta = [-1, -1, +1]
```

Total:

```text
-1 + -1 + 1 = -1
```

Since the total net gas is negative, the circle as a whole loses fuel.

That means:

```text
sum(gas) < sum(cost)
```

So the answer must be:

```text
-1
```

No starting point can overcome a global shortage.

---

### 9. Code

```python
class Solution:
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        start = 0
        current_balance = 0
        total_balance = 0

        for i in range(len(gas)):
            gain = gas[i] - cost[i]
            total_balance += gain
            current_balance += gain

            if current_balance < 0:
                start = i + 1
                current_balance = 0

        return start if total_balance >= 0 else -1
```

Equivalent pseudocode:

```text
start = 0
current_balance = 0
total_balance = 0

for i from 0 to n - 1:
    gain = gas[i] - cost[i]
    total_balance += gain
    current_balance += gain

    if current_balance < 0:
        start = i + 1
        current_balance = 0

if total_balance < 0:
    return -1
else:
    return start
```

---

### 10. Why the Algorithm Is Correct

We need to show two things:

1. If the algorithm returns `-1`, no solution exists.
2. If the algorithm returns `start`, that `start` is valid.

#### Lemma 1: If `total_balance < 0`, no valid start exists

The total amount of fuel collected on one full loop is:

```text
sum(gas)
```

The total amount of fuel required is:

```text
sum(cost)
```

If:

```text
sum(gas) < sum(cost)
```

then the car is missing fuel overall. A full loop is impossible no matter where you start.

So returning `-1` is correct.

#### Lemma 2: When `current_balance` becomes negative at index `i`, no station from the current `start` through `i` can be a valid start

This is exactly the failed-segment argument from Section 4.

If starting at `start` first fails at `i`, then every station inside that segment also fails by or before `i`.

So moving `start` to `i + 1` never discards a valid answer.

#### Lemma 3: If `total_balance >= 0`, the final `start` returned by the scan is valid

During the scan, every time a segment fails, we discard only stations that are provably impossible by Lemma 2.

So after the final reset, the remaining candidate `start` is the only one not eliminated by this reasoning.

From `start` to the end of the array, the running balance never goes negative; otherwise the algorithm would have reset again.

After reaching the end, the tank contains the net gain from `start` to `n - 1`.

Because the total balance over the whole circle is nonnegative, that remaining fuel is enough to pay for the wrapped prefix from `0` to `start - 1`.

So the trip starting at `start` completes the full circuit without the tank becoming negative.

Therefore the returned `start` is correct.

Combining the three lemmas proves the algorithm is correct.

---

### 11. Complexity

- Time: `O(n)` because we scan the arrays once.
- Space: `O(1)` extra space because we store only a few integers.

This is a real improvement over the `O(n^2)` brute-force simulation.

---

### 12. Common Pitfalls

- Forgetting the global feasibility check.
  Even if one segment looks promising, `sum(gas) < sum(cost)` means the answer is definitely `-1`.

- Resetting `start` to `i` instead of `i + 1`.
  If failure happens at index `i`, then station `i` is part of the failed segment and is also impossible.

- Forgetting to reset `current_balance` to `0`.
  Once that segment is discarded, its debt should not be carried into the next candidate.

- Simulating the full route from every station.
  That works, but it misses the whole point of the greedy elimination argument.

- Thinking the best start must have the largest local surplus.
  The answer depends on the entire circular prefix behavior, not just one station's immediate profit.

---

### 13. First-Principles Summary

The core idea is simple once the problem is rewritten correctly.

Each station contributes:

```text
gas[i] - cost[i]
```

You need a starting point whose running total around the circle never becomes negative.

Two facts solve the problem:

1. If the total net gas is negative, no answer exists.
2. If a candidate start fails at station `i`, then every station inside that failed segment also fails, so you can jump directly to `i + 1`.

That is why a single left-to-right pass is enough.

The algorithm is not "greedy" by magic. It works because each failure proves an entire block of candidates is impossible.

## Implementation

See `solutions/array_string/p134_gas_station.py`.

## Tests

See `tests/array_string/test_p134_gas_station.py`.

## Examples

### Example 1
- Input: `{'gas': [1, 2, 3, 4, 5], 'cost': [3, 4, 5, 1, 2]}`
- Output: `3`

### Example 2
- Input: `{'gas': [2, 3, 4], 'cost': [3, 4, 3]}`
- Output: `-1`
