# 53. Maximum Subarray

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/maximum-subarray/
- Official Group: Kadane's Algorithm
- Pattern Group: Kadane
- Patterns: kadane, window-or-prefix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer array:

```text
nums = [a0, a1, a2, ...]
```

find the largest possible sum of a non-empty contiguous subarray.

Contiguous means the elements must appear next to each other in the original array. You may choose where the subarray starts and where it ends, but once you choose those boundaries, you must take everything between them.

For example, in:

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
```

one possible subarray is:

```text
[4, -1, 2, 1]
```

Its sum is:

```text
4 + (-1) + 2 + 1 = 6
```

The problem asks for the maximum sum, not the subarray itself, so the answer for this example is:

```text
6
```

The real problem is:

> Among all non-empty contiguous intervals, find the largest interval sum.

---

### 2. Start From the Brute Force Baseline

The most direct way to solve the problem is to try every possible subarray.

A subarray is determined by two indices:

```text
left  = starting index
right = ending index
```

So we can enumerate every pair `(left, right)` and compute:

```text
nums[left] + nums[left + 1] + ... + nums[right]
```

Conceptually:

```python
best = -infinity

for left in range(len(nums)):
    for right in range(left, len(nums)):
        total = sum(nums[left:right + 1])
        best = max(best, total)
```

This is correct because it checks every candidate answer.

But it is too slow:

- There are `O(n^2)` possible subarrays.
- If each sum is recomputed from scratch, the total time can become `O(n^3)`.

We can improve sum computation with prefix sums and reach `O(n^2)`, but Kadane's algorithm goes further by asking a sharper question:

> When scanning left to right, how much information from the past can still matter for a future best subarray?

---

### 3. The Key Observation

Suppose we are deciding the best subarray that ends exactly at the current index `i`.

That subarray has only two possible shapes:

```text
1. It starts at i.
2. It extends a subarray that ended at i - 1.
```

There is no third option.

If the subarray ends at `i`, then its last element is definitely `nums[i]`. Everything before `nums[i]`, if included, must be a contiguous suffix ending at `i - 1`.

So the decision is:

```text
start fresh at nums[i]
or
extend the best subarray ending at i - 1
```

This gives the recurrence:

```text
best_ending_here = max(nums[i], previous_best_ending_here + nums[i])
```

Why can we keep only the best subarray ending at `i - 1`?

Because if two subarrays both end at `i - 1`, the one with the larger sum is always at least as good to extend by `nums[i]`.

For example, if one suffix has sum `10` and another has sum `3`, then after adding the same next value, the suffix with sum `10` still remains better:

```text
10 + nums[i] >= 3 + nums[i]
```

So all weaker suffixes can be forgotten.

---

### 4. Why Bad Prefixes Should Be Dropped

Another way to see Kadane's algorithm is through the idea of a harmful prefix.

If the best subarray ending before the current number has a negative sum, then carrying it forward can only reduce any future subarray.

For example:

```text
previous best suffix = -4
current number       = 7
```

Extending gives:

```text
-4 + 7 = 3
```

Starting fresh gives:

```text
7
```

So the negative prefix should be discarded.

But if the previous best suffix is positive, it helps:

```text
previous best suffix = 5
current number       = 7
```

Extending gives:

```text
5 + 7 = 12
```

Starting fresh gives:

```text
7
```

So we should extend.

This is the heart of the problem:

> A past suffix is worth keeping only if it improves the sum of a future subarray.

---

### 5. The Kadane Invariant

Maintain two values:

```text
current = maximum sum of any non-empty subarray ending at the current index
best    = maximum sum of any non-empty subarray seen anywhere so far
```

The word ending is crucial.

`current` is not the best answer overall. It is the best answer under the constraint that the subarray must include the current element as its right endpoint.

`best` is the global answer among all positions processed so far.

After processing `nums[i]`, the invariant is:

```text
current is the best subarray sum ending exactly at i
best is the best subarray sum among all subarrays ending at indices 0 through i
```

The update is:

```text
current = max(nums[i], current + nums[i])
best = max(best, current)
```

The first line decides whether to start a new subarray at `i` or extend the best suffix from `i - 1`.

The second line records whether this newly computed suffix is the best subarray seen anywhere.

---

### 6. Algorithm

Because the subarray must be non-empty, initialize from the first element:

```text
current = nums[0]
best = nums[0]
```

Then scan the rest of the array.

For each value `x`:

1. Decide whether the best subarray ending here should start at `x` or extend the previous `current`:

```text
current = max(x, current + x)
```

2. Use that ending-here sum to update the global answer:

```text
best = max(best, current)
```

3. After the scan, return `best`.

This works in one pass because each position only needs the best suffix from the previous position, not the full history of all subarrays.

---

### 7. Detailed Walkthrough

Use the first official example:

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
```

Initialize from the first element:

```text
current = -2
best = -2
```

#### Visit `1`

We compare starting fresh with extending:

```text
start fresh: 1
extend:      -2 + 1 = -1
```

Starting fresh is better:

```text
current = 1
best = max(-2, 1) = 1
```

The best subarray ending here is `[1]`.

#### Visit `-3`

Compare:

```text
start fresh: -3
extend:       1 + (-3) = -2
```

Extending is better, even though the result is negative:

```text
current = -2
best = max(1, -2) = 1
```

The best subarray ending here is `[1, -3]`, with sum `-2`.

The global best remains `[1]`.

#### Visit `4`

Compare:

```text
start fresh: 4
extend:      -2 + 4 = 2
```

Starting fresh is better:

```text
current = 4
best = max(1, 4) = 4
```

The bad suffix ending before `4` is dropped.

#### Visit `-1`

Compare:

```text
start fresh: -1
extend:       4 + (-1) = 3
```

Extending is better:

```text
current = 3
best = max(4, 3) = 4
```

The best subarray ending here is `[4, -1]`.

The global best remains `[4]`.

#### Visit `2`

Compare:

```text
start fresh: 2
extend:      3 + 2 = 5
```

Extending is better:

```text
current = 5
best = max(4, 5) = 5
```

The best subarray ending here is `[4, -1, 2]`.

#### Visit `1`

Compare:

```text
start fresh: 1
extend:      5 + 1 = 6
```

Extending is better:

```text
current = 6
best = max(5, 6) = 6
```

The best subarray ending here is `[4, -1, 2, 1]`.

#### Visit `-5`

Compare:

```text
start fresh: -5
extend:       6 + (-5) = 1
```

Extending is better:

```text
current = 1
best = max(6, 1) = 6
```

The global best remains `[4, -1, 2, 1]`.

#### Visit `4`

Compare:

```text
start fresh: 4
extend:      1 + 4 = 5
```

Extending is better:

```text
current = 5
best = max(6, 5) = 6
```

Final answer:

```text
6
```

The maximum-sum subarray is:

```text
[4, -1, 2, 1]
```

---

### 8. Code

```python
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current = nums[0]
        best = nums[0]

        for value in nums[1:]:
            current = max(value, current + value)
            best = max(best, current)

        return best
```

Equivalent pseudocode:

```text
current = nums[0]
best = nums[0]

for each value after the first:
    current = max(value, current + value)
    best = max(best, current)

return best
```

---

### 9. Why This Code Is Correct

We prove the invariant by induction.

After initialization at index `0`:

```text
current = nums[0]
best = nums[0]
```

There is only one non-empty subarray ending at index `0`: `[nums[0]]`. So `current` is correct.

There is also only one non-empty subarray seen so far, so `best` is correct.

Now assume that after processing index `i - 1`:

```text
current = maximum sum of any non-empty subarray ending at i - 1
best = maximum sum of any non-empty subarray in nums[0:i]
```

At index `i`, any non-empty subarray ending at `i` must either:

```text
1. consist only of nums[i], or
2. be a subarray ending at i - 1 with nums[i] appended
```

By the induction hypothesis, the best possible subarray of the second type has sum:

```text
previous current + nums[i]
```

Therefore the best subarray ending at `i` has sum:

```text
max(nums[i], previous current + nums[i])
```

which is exactly how the algorithm updates `current`.

Once `current` is correct for index `i`, the best subarray seen anywhere from index `0` through `i` is either:

```text
1. the previous global best, or
2. the best subarray ending at i
```

The algorithm updates:

```text
best = max(best, current)
```

so `best` is also correct after index `i`.

By induction, after every index is processed, `best` is the maximum sum among all non-empty contiguous subarrays in the entire array.

That is exactly what the problem asks for.

---

### 10. Complexity

The algorithm scans the array once.

At each element, it performs only constant-time work:

```text
one addition
one max for current
one max for best
```

So the time complexity is:

```text
O(n)
```

It stores only two integer variables, regardless of input size:

```text
current
best
```

So the auxiliary space complexity is:

```text
O(1)
```

---

### 11. Common Pitfalls

#### Pitfall 1: Initializing `best` to `0`

The subarray must be non-empty, and all numbers may be negative.

For example:

```text
nums = [-3, -2, -5]
```

The correct answer is:

```text
-2
```

If `best` starts at `0`, the algorithm may incorrectly return `0`, which corresponds to choosing an empty subarray. Empty subarrays are not allowed.

Initialize from `nums[0]` instead.

#### Pitfall 2: Confusing `current` with `best`

`current` must end at the current index.

`best` can end anywhere seen so far.

In the walkthrough, after processing `-5`, `current` becomes `1`, but `best` remains `6`. That is normal: the best suffix ending at `-5` is not the best subarray overall.

#### Pitfall 3: Resetting at the wrong time

Some versions write Kadane as:

```python
current += value
best = max(best, current)
if current < 0:
    current = 0
```

That style can work when written carefully, but it is easier to make mistakes with all-negative arrays. The recurrence form:

```python
current = max(value, current + value)
```

keeps the non-empty-subarray rule explicit.

#### Pitfall 4: Treating this as a normal sliding window

There is no fixed condition like "sum must be at least target" and no monotonic expand/shrink rule.

Negative numbers mean adding an element can decrease the sum, and removing an element can increase it.

Kadane's algorithm works because it tracks the best suffix ending at each index, not because it maintains a valid window with two moving boundaries.

#### Pitfall 5: Returning the subarray when the problem asks for the sum

The LeetCode problem asks for the maximum sum only.

Tracking boundaries is possible, but not necessary for this problem.

---

### 12. First-Principles Summary

This problem follows from these basic facts:

```text
1. A candidate answer is a non-empty contiguous interval.
2. When scanning left to right, focus on intervals that end at the current index.
3. Any interval ending at the current index either starts here or extends an interval ending at the previous index.
4. Among all previous intervals ending at the previous index, only the largest sum can ever be useful.
5. Therefore one variable, current, is enough to remember the best extendable suffix.
6. A second variable, best, records the best answer seen anywhere.
```

In one sentence:

> At each index, choose whether to start a new subarray or extend the best previous suffix, then use that ending-here result to update the global maximum.

## Implementation
See `solutions/kadane/p053_maximum_subarray.py`.

## Tests
See `tests/kadane/test_p053_maximum_subarray.py`.

## Examples

### Example 1
- Input: `{'nums': [-2, 1, -3, 4, -1, 2, 1, -5, 4]}`
- Output: `6`

### Example 2
- Input: `{'nums': [1]}`
- Output: `1`

### Example 3
- Input: `{'nums': [5, 4, -1, 7, 8]}`
- Output: `23`

## Follow-up Practice
- Trace an all-negative input such as `[-3, -2, -5]` and verify that the answer is `-2`.
- Explain why `current` means "best subarray ending here," not "best subarray overall."
- Modify the algorithm to also return the start and end indices of the maximum subarray.
