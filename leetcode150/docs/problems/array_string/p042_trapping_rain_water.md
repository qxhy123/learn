# 42. Trapping Rain Water

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/trapping-rain-water/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: two-pointers, prefix-maximum, greedy

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `height` where each number is the height of a vertical bar of width `1`.

For example:

```text
height = [0, 1, 0, 2]
```

means:

```text
index:   0  1  2  3
height:  0  1  0  2
```

Imagine rain falling on top of these bars.

Water can sit in the valleys between taller bars, but only if there is:

- a wall on the left
- a wall on the right

The question is:

> After raining, how many total unit squares of water remain trapped?

This is not asking for the highest pool, or the water above one index, or a simulation of water flowing one drop at a time.

It is asking for the sum over all positions:

```text
water at index i
```

So the real problem is:

> For each position, determine how much water can stand above that bar, then add those amounts together.

---

### 2. What Determines the Water Above One Bar?

Pick one index `i`.

Suppose the tallest wall to its left has height:

```text
left_max
```

and the tallest wall to its right has height:

```text
right_max
```

Then the water above index `i` is:

```text
min(left_max, right_max) - height[i]
```

if that value is positive, otherwise `0`.

Why `min(left_max, right_max)`?

Because water spills over the shorter side.

If the left side reaches height `5` and the right side reaches height `2`, then water cannot be held above level `2`, no matter how tall the left wall is.

So each index is governed by a simple local law:

> Water level is limited by the shorter of the best wall on the left and the best wall on the right.

That formula is the entire problem.

The rest of the algorithm is about computing it efficiently.

---

### 3. Start From the Brute-Force Idea

The most direct method is:

1. For each index `i`, scan left to find the tallest bar in `height[0..i]`.
2. Scan right to find the tallest bar in `height[i..n-1]`.
3. Compute:

```text
water[i] = min(left_max, right_max) - height[i]
```

4. Add the positive amounts.

Conceptually:

```python
total = 0

for i in range(n):
    left_max = max(height[0:i + 1])
    right_max = max(height[i:n])
    total += max(0, min(left_max, right_max) - height[i])
```

This is correct, but expensive.

For every index, you may scan almost the whole array to the left and almost the whole array to the right.

That gives:

- Time: `O(n^2)`
- Space: `O(1)` extra

The brute-force solution is useful because it reveals the actual dependency:

```text
each index needs the best wall seen from the left
and the best wall seen from the right
```

That observation leads to better solutions.

---

### 4. A Better Baseline: Precompute Left and Right Maxima

Instead of recomputing those maxima for every index, we can store them.

Define:

```text
left_max[i]  = tallest bar in height[0..i]
right_max[i] = tallest bar in height[i..n-1]
```

Then:

```text
water at i = min(left_max[i], right_max[i]) - height[i]
```

This removes the repeated scanning.

How to build the arrays:

- Sweep left to right to fill `left_max`
- Sweep right to left to fill `right_max`

Then sweep once more to accumulate water.

This gives:

- Time: `O(n)`
- Space: `O(n)`

This version is already clean and fully efficient in time.

But the standard interview follow-up is:

> Can we get rid of the extra arrays?

Yes, if we understand exactly when one side is already sufficient to decide the water at the current position.

---

### 5. The Key Observation Behind the Two-Pointer Solution

Suppose we keep two pointers:

```text
left  at the start
right at the end
```

and also track:

```text
left_max  = tallest bar seen so far from the left
right_max = tallest bar seen so far from the right
```

Now compare `left_max` and `right_max`.

#### Case 1: `left_max <= right_max`

Then the current `left` position already has enough support on the right.

Why?

Because somewhere on the right side there is a wall of height at least `right_max`, and `right_max >= left_max`.

So for the current `left` index, the limiting side cannot be the right side anymore.
The shorter side is `left_max`.

Therefore the trapped water at `left` is fully determined:

```text
water at left = left_max - height[left]
```

if positive.

We do not need to know the exact future shape of the interior.
We only need to know that the right side is tall enough to avoid being the bottleneck.

#### Case 2: `right_max < left_max`

By symmetric reasoning, the trapped water at `right` is fully determined:

```text
water at right = right_max - height[right]
```

if positive.

This is the heart of the algorithm:

> Always process the side whose running maximum is smaller.

That side is already limited by its own maximum, so its answer is final.

---

### 6. The Invariant

The two-pointer method works because it maintains a precise state:

- `left` and `right` bound the unexplored middle region
- `left_max` is the tallest bar in `height[0..left]`
- `right_max` is the tallest bar in `height[right..n-1]`
- every index strictly outside `[left, right]` has already had its final trapped water added exactly once

At each step:

- if `left_max <= right_max`, we can finalize index `left`
- otherwise, we can finalize index `right`

Once an index is finalized, we never need to revisit it.

That is why one pass is enough.

---

### 7. Detailed Algorithm

1. Initialize:

```text
left = 0
right = n - 1
left_max = 0
right_max = 0
total = 0
```

2. While `left < right`:

- Update `left_max = max(left_max, height[left])`
- Update `right_max = max(right_max, height[right])`

3. Compare the two running maxima.

- If `left_max <= right_max`:
  - the left side is the bottleneck
  - the water at `left` is `left_max - height[left]`
  - add that amount to `total`
  - move `left += 1`

- Otherwise:
  - the right side is the bottleneck
  - the water at `right` is `right_max - height[right]`
  - add that amount to `total`
  - move `right -= 1`

4. When the pointers meet, every index has been processed.

Two important details:

- We update `left_max` and `right_max` before computing trapped water.
  Otherwise we might subtract from an outdated wall height.
- We add only nonnegative amounts, but after updating the maxima, the subtraction is naturally nonnegative on the processed side.

---

### 8. Walk Through the Official Example

Consider:

```text
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
```

The final answer is `6`, but it is worth seeing why.

#### First, index-by-index intuition

If you compute the best walls on each side, you get:

```text
index:      0  1  2  3  4  5  6  7  8  9 10 11
height:     0  1  0  2  1  0  1  3  2  1  2  1
left_max:   0  1  1  2  2  2  2  3  3  3  3  3
right_max:  3  3  3  3  3  3  3  3  2  2  2  1
water:      0  0  1  0  1  2  1  0  0  1  0  0
```

Summing the last row gives:

```text
1 + 1 + 2 + 1 + 1 = 6
```

The two-pointer algorithm reaches the same result without storing both arrays.

#### Now trace the two-pointer process

Start:

```text
left = 0, right = 11
left_max = 0, right_max = 0
total = 0
```

We update the maxima at the current ends:

```text
left_max = max(0, height[0])  = 0
right_max = max(0, height[11]) = 1
```

Since:

```text
left_max <= right_max
```

the left side can be finalized.

At index `0`:

```text
water = left_max - height[0] = 0 - 0 = 0
```

Move `left` to `1`.

---

At `left = 1`:

```text
left_max = max(0, 1) = 1
right_max = 1
```

Again `left_max <= right_max`, so finalize the left side:

```text
water at 1 = 1 - 1 = 0
total = 0
```

Move `left` to `2`.

---

At `left = 2`:

```text
left_max = max(1, 0) = 1
right_max = 1
```

Still `left_max <= right_max`, so:

```text
water at 2 = 1 - 0 = 1
total = 1
```

Move `left` to `3`.

---

At `left = 3`:

```text
left_max = max(1, 2) = 2
right_max = 1
```

Now:

```text
left_max > right_max
```

So the right side is the limiting side, and we must finalize `right`.

At index `11`:

```text
water at 11 = 1 - 1 = 0
```

Move `right` to `10`.

---

At `right = 10`:

```text
right_max = max(1, 2) = 2
left_max = 2
```

Now `left_max <= right_max`, so finalize the left side:

```text
water at 3 = 2 - 2 = 0
```

Move `left` to `4`.

---

At `left = 4`:

```text
left_max = 2
right_max = 2
water at 4 = 2 - 1 = 1
total = 2
```

Move `left` to `5`.

---

At `left = 5`:

```text
water at 5 = 2 - 0 = 2
total = 4
```

Move `left` to `6`.

---

At `left = 6`:

```text
water at 6 = 2 - 1 = 1
total = 5
```

Move `left` to `7`.

---

At `left = 7`:

```text
left_max = max(2, 3) = 3
right_max = 2
```

Now the right side is smaller, so finalize from the right.

At `right = 10`:

```text
water at 10 = 2 - 2 = 0
```

Move `right` to `9`.

---

At `right = 9`:

```text
right_max = max(2, 1) = 2
left_max = 3
water at 9 = 2 - 1 = 1
total = 6
```

Move `right` to `8`.

---

At `right = 8`:

```text
water at 8 = 2 - 2 = 0
```

Move `right` to `7`.

Now `left == right`, so we stop.

Total trapped water:

```text
6
```

The important point is not the mechanics of the trace.
The important point is why each processed side was safe to finalize:

- when the left running max was smaller, the left index depended only on that left max
- when the right running max was smaller, the right index depended only on that right max

---

### 9. Pseudocode

```python
def trap(height):
    left = 0
    right = len(height) - 1
    left_max = 0
    right_max = 0
    total = 0

    while left < right:
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        if left_max <= right_max:
            total += left_max - height[left]
            left += 1
        else:
            total += right_max - height[right]
            right -= 1

    return total
```

If you prefer, you can think of this as the `O(n)`-space prefix/suffix solution compressed into a constant-space scan.

---

### 10. Why This Is Correct

We prove the algorithm by showing that every step adds the correct water for exactly one index.

#### When `left_max <= right_max`

For the current `left` index:

- the tallest wall on the left side is `left_max`
- there exists a wall on the right side with height at least `right_max`
- and `right_max >= left_max`

So the shorter of the two bounding sides is definitely `left_max`.

That means the true water above index `left` is:

```text
min(true_left_max, true_right_max) - height[left] = left_max - height[left]
```

So adding `left_max - height[left]` is correct, and we can move `left` forward permanently.

#### When `right_max < left_max`

The argument is symmetric.

For the current `right` index, the shorter bounding side is definitely `right_max`, so the water above that index is:

```text
right_max - height[right]
```

and we can move `right` inward permanently.

#### No index is missed or double-counted

Every loop iteration moves exactly one pointer inward.
So each index is processed once, and only once.

Because each processed amount is correct at the moment it is added, the final total is correct.

---

### 11. Complexity

#### Brute force

- Time: `O(n^2)`
- Space: `O(1)`

#### Prefix/suffix maxima baseline

- Time: `O(n)`
- Space: `O(n)`

#### Two-pointer solution

- Time: `O(n)`
- Space: `O(1)`

Each pointer moves inward at most `n` times total, so the full scan is linear.

---

### 12. Common Pitfalls

#### 1. Using the current bar heights instead of the running maxima

Comparing `height[left]` and `height[right]` is not enough by itself for this formulation.
The decisive quantities are the best walls seen so far:

```text
left_max and right_max
```

Those are what determine whether one side is already safe to finalize.

#### 2. Forgetting that edge bars cannot hold water by themselves

An edge may be tall, but without a boundary on both sides, no water is trapped there.
The formulas naturally handle this, but it helps to remember it when tracing examples.

#### 3. Subtracting before updating the maxima

If you compute water first and only then update `left_max` or `right_max`, you can get negative or outdated results.
The running maxima must reflect the current endpoints before you finalize either side.

#### 4. Assuming a taller far wall always helps immediately

For a position to hold water, both sides matter, and the shorter side is always the limit.
One gigantic wall cannot compensate for a short wall on the other side.

#### 5. Mixing the `O(n)`-space and `O(1)`-space ideas

The prefix/suffix approach says:

```text
compute exact left_max[i] and right_max[i] for every i
```

The two-pointer approach says:

```text
only process the side whose answer is already forced
```

Both are correct, but they rely on slightly different reasoning.

---

### 13. First-Principles Summary

This problem looks geometric, but its logic is simple:

1. Water above an index is determined by the tallest wall on the left and the tallest wall on the right.
2. The usable water level is the smaller of those two walls.
3. A direct implementation scans outward from every index, but repeats work.
4. Precomputing left and right maxima removes the repeated work.
5. The two-pointer optimization goes one step further:
   when one side's running maximum is smaller, that side's current index is already fully determined.

So the deep idea is not "memorize a two-pointer trick."

It is:

> Once one side is known to be the bottleneck, the current bar on that side can be solved immediately.

That is why the algorithm is linear, constant-space, and correct.

## Implementation

See `solutions/array_string/p042_trapping_rain_water.py`.

## Tests

See `tests/array_string/test_p042_trapping_rain_water.py`.

## Examples

### Example 1
- Input: `{'height': [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]}`
- Output: `6`

### Example 2
- Input: `{'height': [4, 2, 0, 3, 2, 5]}`
- Output: `9`
