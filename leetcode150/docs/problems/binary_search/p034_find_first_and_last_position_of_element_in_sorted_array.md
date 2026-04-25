# 34. Find First and Last Position of Element in Sorted Array

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a sorted array `nums` and a value `target`.

The array may contain duplicates. If `target` appears, all of its copies must be in one contiguous block because the array is sorted.

For example:

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8
```

The `8`s occupy this block:

```text
index:   0  1  2  3  4   5
nums:   [5, 7, 7, 8, 8, 10]
                  ^  ^
answer = [3, 4]
```

So the problem is not asking whether `target` exists. It is asking for both boundaries of the block of values equal to `target`:

```text
first position where nums[i] == target
last  position where nums[i] == target
```

If the block does not exist, return:

```text
[-1, -1]
```

The important constraint is that the required running time is `O(log n)`, so a linear scan is not enough for the intended solution.

---

### 2. Start From the Brute Force Idea

The most direct solution is to scan the whole array and remember the first and last index where `target` appears.

Conceptually:

```python
first = -1
last = -1

for i, value in enumerate(nums):
    if value == target:
        if first == -1:
            first = i
        last = i

return [first, last]
```

This is correct because it examines every possible position.

But it costs:

```text
O(n) time
```

That wastes the sorted order. In a sorted array, one comparison tells us not just about one index, but about a whole side of the array.

The deeper question is:

> Can we find the two boundaries without inspecting every element?

Yes. Each boundary is a binary-search boundary problem.

---

### 3. Why a Normal Binary Search Is Not Enough

A standard binary search can find some occurrence of `target`.

For example, in:

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8
```

A normal binary search might find index `3` or index `4`.

But the problem needs:

```text
[3, 4]
```

Finding any match does not tell us whether there are more copies immediately to the left or right.

One possible fix is:

1. Binary search for any `target`.
2. Scan left while values are still `target`.
3. Scan right while values are still `target`.

That can degrade to `O(n)` when the whole array is equal to `target`:

```text
nums   = [8, 8, 8, 8, 8, 8]
target = 8
```

So we need binary search to find boundaries directly, not just a matching element.

---

### 4. The Key Observation: Convert Equality Into Boundaries

Because the array is sorted, all values fall into three regions relative to `target`:

```text
values < target     values == target     values > target
```

For example:

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8

          < target     == target   > target
nums   = [5, 7, 7,      8, 8,       10]
index     0  1  2       3  4         5
```

The first occurrence of `target` is the first index where the value is at least `target`:

```text
first index i such that nums[i] >= target
```

For the example, that is index `3`.

The position after the last occurrence is the first index where the value is greater than `target`:

```text
first index i such that nums[i] > target
```

For the example, that is index `5`.

Therefore:

```text
left_boundary  = first index with nums[i] >= target
right_boundary = first index with nums[i] > target
answer         = [left_boundary, right_boundary - 1]
```

This is the central trick.

Instead of searching for equality, search for two partition points.

---

### 5. The Boundary Search Invariant

Use a half-open search interval:

```text
[left, right)
```

That means:

```text
left is included
right is excluded
```

Initially:

```text
left = 0
right = len(nums)
```

So the candidate range is the entire array.

For a boundary search, define a predicate that changes from `False` to `True` exactly once.

For the left boundary:

```text
predicate(i): nums[i] >= target
```

Because the array is sorted:

```text
False, False, False, True, True, True
```

For the right boundary:

```text
predicate(i): nums[i] > target
```

Again, because the array is sorted:

```text
False, False, False, False, False, True
```

The invariant is:

```text
All indices before left are known to be False.
All indices at or after right are known to be True.
The first True, if it exists, is somewhere in [left, right).
```

At each step:

```text
mid = (left + right) // 2
```

If `predicate(mid)` is true, then `mid` could be the first true index, but everything to the right of `mid` is not needed to find the first true. Move the right boundary:

```text
right = mid
```

If `predicate(mid)` is false, then `mid` and everything before it cannot be the first true. Move the left boundary past `mid`:

```text
left = mid + 1
```

When the loop ends:

```text
left == right
```

That single position is the first index where the predicate becomes true. If no array element satisfies the predicate, the position is `len(nums)`.

---

### 6. Detailed Algorithm

We can write one helper function:

```text
lower_bound(condition)
```

In this problem, it is easier to think of it as:

```text
first_index_at_least(x)
```

It returns the first index `i` where:

```text
nums[i] >= x
```

Then:

```text
start = first_index_at_least(target)
end_exclusive = first_index_at_least(target + 1)
```

That works for integer arrays, but a more general version avoids relying on `target + 1` and searches directly for `nums[i] > target`.

So the robust plan is:

1. Find `left_bound`, the first index where `nums[i] >= target`.
2. If `left_bound == len(nums)`, `target` is larger than every value, so return `[-1, -1]`.
3. If `nums[left_bound] != target`, the first value at least `target` is actually greater than `target`, so `target` is absent. Return `[-1, -1]`.
4. Find `right_bound`, the first index where `nums[i] > target`.
5. Return `[left_bound, right_bound - 1]`.

The two searches differ only in the comparison:

```text
first boundary: nums[mid] >= target
second boundary: nums[mid] > target
```

---

### 7. Pseudocode

```python
def searchRange(nums, target):
    def first_at_least_target():
        left = 0
        right = len(nums)

        while left < right:
            mid = (left + right) // 2

            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1

        return left

    def first_greater_than_target():
        left = 0
        right = len(nums)

        while left < right:
            mid = (left + right) // 2

            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1

        return left

    start = first_at_least_target()

    if start == len(nums) or nums[start] != target:
        return [-1, -1]

    end_exclusive = first_greater_than_target()
    return [start, end_exclusive - 1]
```

You may also combine both helpers into one function that accepts a comparison or a value, but keeping the two searches explicit is often easier to understand when first learning boundary binary search.

---

### 8. Walk Through Example 1

Input:

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8
```

First search: find the first index where `nums[i] >= 8`.

Initial state:

```text
left = 0
right = 6
candidate interval = [0, 6)
```

Step 1:

```text
mid = (0 + 6) // 2 = 3
nums[3] = 8
nums[3] >= 8 is True
```

Index `3` might be the first `8`, but there could be another `8` to its left, so keep the left half including `mid`:

```text
right = 3
candidate interval = [0, 3)
```

Step 2:

```text
mid = (0 + 3) // 2 = 1
nums[1] = 7
nums[1] >= 8 is False
```

Index `1` and everything before it are too small, so discard them:

```text
left = 2
candidate interval = [2, 3)
```

Step 3:

```text
mid = (2 + 3) // 2 = 2
nums[2] = 7
nums[2] >= 8 is False
```

Discard index `2`:

```text
left = 3
candidate interval = [3, 3)
```

The loop stops. The first index where `nums[i] >= 8` is:

```text
start = 3
```

Now verify:

```text
nums[3] == 8
```

So `target` exists.

Second search: find the first index where `nums[i] > 8`.

Initial state again:

```text
left = 0
right = 6
```

Step 1:

```text
mid = 3
nums[3] = 8
nums[3] > 8 is False
```

Index `3` is not greater than `target`, so the first greater value must be after it:

```text
left = 4
```

Step 2:

```text
mid = (4 + 6) // 2 = 5
nums[5] = 10
nums[5] > 8 is True
```

Index `5` might be the first greater value, so keep the left side including `5`:

```text
right = 5
```

Step 3:

```text
mid = (4 + 5) // 2 = 4
nums[4] = 8
nums[4] > 8 is False
```

Index `4` is still equal to target, so the first greater value is after it:

```text
left = 5
```

The loop stops:

```text
right_bound = 5
```

This is the first index after the block of `8`s. Therefore the last `8` is:

```text
right_bound - 1 = 4
```

Final answer:

```text
[3, 4]
```

---

### 9. Walk Through Missing Target

Input:

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 6
```

Find the first index where `nums[i] >= 6`.

The answer is index `1` because:

```text
nums[1] = 7
```

But after the search, we check:

```text
nums[1] == 6
```

This is false.

That means `6` would be inserted at index `1`, between `5` and `7`, but it is not actually present.

So return:

```text
[-1, -1]
```

This verification step is essential. A lower-bound search returns where the target should start, not proof that the target exists.

---

### 10. Correctness

We prove that the algorithm returns the first and last positions of `target` if `target` appears, and `[-1, -1]` otherwise.

#### Boundary Search Lemma

Consider the helper search for the first index where a monotonic predicate becomes true.

The loop maintains this invariant:

```text
All indices before left are known not to satisfy the predicate.
All indices at or after right are not needed because the first satisfying index is no later than right.
The first satisfying index, if one exists, remains inside [left, right].
```

More concretely for the half-open interval `[left, right)`:

- If `predicate(mid)` is true, then `mid` is a valid candidate for the first true index. Since we are looking for the first such index, no index after `mid` is needed, so setting `right = mid` preserves the answer.
- If `predicate(mid)` is false, then `mid` cannot be the first true index, and by monotonicity every index before `mid` is also false. Setting `left = mid + 1` preserves the answer.

Each update strictly shrinks the interval, so the loop terminates. When it terminates, `left == right`, and by the invariant this position is exactly the first index where the predicate is true, or `len(nums)` if no such index exists.

#### Applying the Lemma to the Left Boundary

For the first search, the predicate is:

```text
nums[i] >= target
```

Because `nums` is sorted, once this predicate is true at some index, it remains true for every later index.

By the lemma, the search returns the first index whose value is at least `target`.

If this index is outside the array, then no value is at least `target`, so `target` cannot appear.

If this index is inside the array but `nums[index] != target`, then `nums[index] > target`. Since it is the first value at least `target`, all earlier values are smaller than `target`, and this value is already greater than `target`. Therefore no value equals `target`.

In both cases, returning `[-1, -1]` is correct.

Otherwise, `nums[index] == target`, and because it is the first value at least `target`, it is the first occurrence of `target`.

#### Applying the Lemma to the Right Boundary

For the second search, the predicate is:

```text
nums[i] > target
```

Because `nums` is sorted, once this predicate is true at some index, it remains true for every later index.

By the lemma, the search returns the first index whose value is greater than `target`.

Every index before that boundary has value less than or equal to `target`. Since we already know a block of `target` values exists, the last target must be immediately before the first greater value.

Therefore:

```text
last = right_boundary - 1
```

is the last occurrence of `target`.

Combining the proven first occurrence and last occurrence, the algorithm returns exactly the required range.

---

### 11. Complexity

Each boundary search halves the remaining interval on every iteration.

For an array of length `n`, one search costs:

```text
O(log n)
```

The algorithm performs two such searches, so the total time is:

```text
O(log n) + O(log n) = O(log n)
```

It uses only a few index variables, so the extra space is:

```text
O(1)
```

---

### 12. Common Pitfalls

- Stopping when `nums[mid] == target`. That finds one occurrence, not necessarily the first or last.
- Scanning outward after finding a match. This can become `O(n)` when many elements equal `target`.
- Forgetting to verify `nums[start] == target` after the left-bound search.
- Accessing `nums[start]` before checking `start == len(nums)`.
- Mixing inclusive and half-open interval rules. If the loop uses `[left, right)`, initialize `right = len(nums)` and loop while `left < right`.
- Updating `right = mid - 1` in a half-open boundary search. That belongs to a different inclusive-interval style and can skip the answer.
- Computing the last index as `right_bound` instead of `right_bound - 1`.
- Using `target + 1` to find the right boundary in a language or setting where overflow or non-integer values might matter. Searching for `nums[mid] > target` avoids that issue.
- Returning `[start, start]` for a single occurrence without proving the right boundary. Single occurrence works naturally as `right_bound - 1 == start`.

---

### 13. First-Principles Summary

The sorted array turns equal values into one contiguous block.

So the range of `target` is determined by two partition points:

```text
first index where value >= target
first index where value > target
```

Binary search works because each comparison classifies an entire half of the remaining interval as unable to contain the boundary.

The invariant is the core idea:

```text
keep the first possible true position inside the current interval
```

Once the left boundary is found, verify that the target actually exists. Once the right boundary is found, subtract one to get the last target index.

That gives the target range in logarithmic time without ever scanning through the duplicate block.

## Implementation
See `solutions/binary_search/p034_find_first_and_last_position_of_element_in_sorted_array.py`.

## Tests
See `tests/binary_search/test_p034_find_first_and_last_position_of_element_in_sorted_array.py`.

## Examples

### Example 1
- Input: `{'nums': [5, 7, 7, 8, 8, 10], 'target': 8}`
- Output: `[3, 4]`

### Example 2
- Input: `{'nums': [5, 7, 7, 8, 8, 10], 'target': 6}`
- Output: `[-1, -1]`

### Example 3
- Input: `{'nums': [], 'target': 0}`
- Output: `[-1, -1]`

## Follow-up Practice
- Trace both boundary searches on `[1]` with targets `0`, `1`, and `2`.
- Rewrite the solution using one reusable `first_true(predicate)` helper.
- Implement the same idea with inclusive `[left, right]` bounds, then compare the invariants.
- Explain why the answer is still `O(log n)` when every element equals `target`.
