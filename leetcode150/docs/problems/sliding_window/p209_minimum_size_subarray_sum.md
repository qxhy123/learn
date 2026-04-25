# 209. Minimum Size Subarray Sum

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/minimum-size-subarray-sum/
- Official Group: Sliding Window
- Pattern Group: Sliding Window
- Patterns: sliding-window, window-or-prefix, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given:

```text
target = a positive integer
nums   = an array of positive integers
```

Find the minimum length of a contiguous subarray whose sum is at least `target`.

If no such subarray exists, return `0`.

For example:

```text
target = 7
nums = [2, 3, 1, 2, 4, 3]
```

The subarray:

```text
[2, 3, 1, 2]
```

has sum:

```text
2 + 3 + 1 + 2 = 8
```

So it is valid, with length `4`.

But the subarray:

```text
[4, 3]
```

has sum:

```text
4 + 3 = 7
```

It is also valid, and its length is only `2`.

So the answer is:

```text
2
```

The real problem is:

> Among all contiguous intervals whose sum is at least `target`, find the shortest length.

---

### 2. Start From the Brute Force Idea

The most direct approach is:

1. Enumerate every starting index `left`.
2. Enumerate every ending index `right`.
3. Compute the sum of `nums[left:right + 1]`.
4. If the sum is at least `target`, update the shortest length.

Conceptually:

```python
best = infinity

for left in range(len(nums)):
    for right in range(left, len(nums)):
        total = sum(nums[left:right + 1])
        if total >= target:
            best = min(best, right - left + 1)
```

This is correct, but inefficient.

There are `O(n^2)` subarrays, and recomputing sums can make it even slower.

We can improve the sum calculation with prefix sums, but there is an even more important first-principles observation:

> All numbers are positive.

That single fact gives us monotonic behavior.

---

### 3. Why Positivity Matters

Because every number in `nums` is positive:

```text
Adding a number to the right always increases the window sum.
Removing a number from the left always decreases the window sum.
```

This means the sum behaves predictably as the window changes.

If the current sum is too small:

```text
current_sum < target
```

then removing elements from the left cannot help, because it would only make the sum smaller.

So the only useful action is:

```text
expand right
```

If the current sum is large enough:

```text
current_sum >= target
```

then the current window is valid.

But we want the shortest valid window, so expanding right would only make the window longer. The useful action is:

```text
shrink left
```

to see whether the window can remain valid with fewer elements.

This is the core reason sliding window works here.

---

### 4. A Subarray Is a Window

Any candidate subarray is a contiguous interval:

```text
nums[left:right + 1]
```

So we maintain:

```text
left        = start of the current window
right       = end of the current window
current_sum = sum of nums[left:right + 1]
```

The window may be invalid or valid.

Invalid means:

```text
current_sum < target
```

Valid means:

```text
current_sum >= target
```

The algorithm alternates between two actions:

```text
expand right until valid
shrink left while valid
```

---

### 5. The Window Invariant

Maintain this invariant:

```text
current_sum is exactly the sum of nums[left:right + 1]
```

After adding `nums[right]`:

```text
current_sum += nums[right]
```

After removing `nums[left]`:

```text
current_sum -= nums[left]
left += 1
```

As long as this invariant is true, we can test validity in constant time:

```text
current_sum >= target
```

---

### 6. Algorithm

1. Set:

```text
left = 0
current_sum = 0
best = infinity
```

2. Move `right` from `0` to `len(nums) - 1`.

3. Add the new number:

```text
current_sum += nums[right]
```

4. While the current window is valid:

```text
current_sum >= target
```

record its length:

```text
best = min(best, right - left + 1)
```

then shrink from the left:

```text
current_sum -= nums[left]
left += 1
```

5. If `best` is still infinity, return `0`. Otherwise return `best`.

---

### 7. Example: `target = 7`, `nums = [2, 3, 1, 2, 4, 3]`

Start:

```text
left = 0
current_sum = 0
best = infinity
```

#### Add `2`

```text
window = [2]
current_sum = 2
```

The sum is below `7`, so expand.

#### Add `3`

```text
window = [2, 3]
current_sum = 5
```

Still below `7`, so expand.

#### Add `1`

```text
window = [2, 3, 1]
current_sum = 6
```

Still below `7`, so expand.

#### Add `2`

```text
window = [2, 3, 1, 2]
current_sum = 8
```

Now the window is valid.

Record its length:

```text
best = 4
```

Now try to shrink:

```text
remove 2
window = [3, 1, 2]
current_sum = 6
```

Now invalid, so stop shrinking and expand again.

#### Add `4`

```text
window = [3, 1, 2, 4]
current_sum = 10
```

Valid. Record length:

```text
best = min(4, 4) = 4
```

Shrink:

```text
remove 3
window = [1, 2, 4]
current_sum = 7
```

Still valid. Record length:

```text
best = min(4, 3) = 3
```

Shrink again:

```text
remove 1
window = [2, 4]
current_sum = 6
```

Invalid. Stop shrinking.

#### Add `3`

```text
window = [2, 4, 3]
current_sum = 9
```

Valid. Record length:

```text
best = min(3, 3) = 3
```

Shrink:

```text
remove 2
window = [4, 3]
current_sum = 7
```

Still valid. Record length:

```text
best = min(3, 2) = 2
```

Shrink again:

```text
remove 4
window = [3]
current_sum = 3
```

Invalid. Stop.

Final answer:

```text
2
```

---

### 8. Code

```python
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        current_sum = 0
        best = float("inf")

        for right, value in enumerate(nums):
            current_sum += value

            while current_sum >= target:
                best = min(best, right - left + 1)
                current_sum -= nums[left]
                left += 1

        if best == float("inf"):
            return 0

        return best
```

---

### 9. Why This Code Is Correct

The algorithm maintains this invariant:

```text
current_sum equals the sum of nums[left:right + 1]
```

Every time `right` moves forward, the new element is added to `current_sum`.

Every time `left` moves forward, the old leftmost element is removed from `current_sum`.

So the invariant is always preserved.

Now consider any fixed `right`.

After adding `nums[right]`, if the window sum is less than `target`, then no shorter window ending at this same `right` can be valid by moving `left` forward, because all numbers are positive and removing elements only decreases the sum.

If the window sum is at least `target`, the window is valid. The algorithm records its length, then repeatedly moves `left` forward while the window remains valid. This checks all valid windows ending at this `right` that can be obtained by shrinking from the left.

The moment the window becomes invalid, further shrinking would only decrease the sum even more, so there is no valid shorter window ending at this `right` left to check.

Since the algorithm repeats this process for every possible `right`, it considers every candidate that could be the shortest valid subarray.

Because it records only valid windows and keeps the minimum length among them, the final answer is the minimum size of a subarray whose sum is at least `target`.

If no valid window is ever recorded, then no such subarray exists, and the algorithm correctly returns `0`.

---

### 10. Why It Is `O(n)`

Although the code has a `while` loop inside a `for` loop, it is not `O(n^2)`.

The reason is that both pointers only move forward:

```text
right moves from 0 to n - 1
left moves from 0 to n - 1
```

Each element is:

```text
added to current_sum at most once
removed from current_sum at most once
```

So the total number of pointer moves is at most `2n`.

Complexity:

```text
Time:  O(n)
Space: O(1)
```

---

### 11. Why This Fails With Negative Numbers

The sliding window logic depends on positivity.

If negative numbers are allowed, this statement is no longer true:

```text
Adding to the right always increases the sum.
```

For example:

```text
nums = [5, -10, 20]
target = 15
```

Adding `-10` decreases the sum.

Also, this statement is no longer true:

```text
Removing from the left always decreases the sum.
```

Removing a negative number can increase the sum.

So the simple expand/shrink decision breaks.

For arrays with negative numbers, you usually need a different technique, such as prefix sums with a monotonic deque.

---

### 12. Common Pitfalls

#### Pitfall 1: Using `if` instead of `while`

Once the window is valid, we must shrink as much as possible.

Wrong pattern:

```python
if current_sum >= target:
    ...
```

Correct pattern:

```python
while current_sum >= target:
    ...
```

Using `if` may miss a shorter valid window.

#### Pitfall 2: Updating `best` after shrinking

You should record the current valid window before removing `nums[left]`.

Correct order:

```python
best = min(best, right - left + 1)
current_sum -= nums[left]
left += 1
```

If you remove first, the window may no longer be valid.

#### Pitfall 3: Forgetting the no-answer case

If no window reaches `target`, return:

```text
0
```

not infinity.

#### Pitfall 4: Applying this directly to arrays with negative numbers

This exact sliding window proof requires positive numbers.

Without positivity, the monotonic reasoning is invalid.

---

### 13. First-Principles Summary

This problem follows from these basic facts:

```text
1. A subarray is a contiguous interval.
2. A contiguous interval can be represented by left and right boundaries.
3. The array contains only positive integers.
4. Therefore, expanding right can only increase the sum.
5. Shrinking left can only decrease the sum.
6. If the sum is too small, we must expand.
7. If the sum is large enough, we should shrink to search for a shorter valid interval.
```

In one sentence:

> Use a moving window whose sum is maintained incrementally: expand until the sum reaches `target`, then shrink while it remains valid, recording the shortest valid length seen.

## Implementation

See `solutions/sliding_window/p209_minimum_size_subarray_sum.py`.

## Tests

See `tests/sliding_window/test_p209_minimum_size_subarray_sum.py`.

## Examples

### Example 1
- Input: `{'target': 7, 'nums': [2, 3, 1, 2, 4, 3]}`
- Output: `2`

### Example 2
- Input: `{'target': 4, 'nums': [1, 4, 4]}`
- Output: `1`

### Example 3
- Input: `{'target': 11, 'nums': [1, 1, 1, 1, 1, 1, 1, 1]}`
- Output: `0`

## Follow-up Practice
- Trace why the inner loop must be `while`, not `if`.
- Compare this positive-only window with a prefix-sum approach.
- Study why negative numbers require a different technique such as a monotonic deque.
