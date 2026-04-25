# 35. Search Insert Position

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/search-insert-position/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### What The Problem Is Asking
You are given a sorted array of distinct integers, `nums`, and a `target` value. The task is to return the index where `target` belongs.

There are two cases:

- If `target` already appears in `nums`, return its current index.
- If `target` does not appear, return the index where inserting it would keep `nums` sorted.

So the answer is not merely “find the target.” It is the first position where the array value is greater than or equal to `target`.

In other words, we want the boundary between these two regions:

```text
values < target | values >= target
```

The returned index is the start of the right region. If every number is smaller than `target`, that right region is empty and the answer is `len(nums)`. If the first number is already greater than or equal to `target`, the answer is `0`.

### Baseline: Linear Scan
The most direct solution is to scan from left to right:

```text
for each index i:
    if nums[i] >= target:
        return i
return len(nums)
```

This works because the array is sorted. The first time we see a value not less than `target`, every later value is also not less than `target`, so `i` is exactly where `target` belongs.

For example, with `nums = [1, 3, 5, 6]` and `target = 2`, we scan:

```text
1 < 2, keep going
3 >= 2, return index 1
```

The baseline is simple and correct, but it may inspect every element. Its time complexity is `O(n)`. Since the input is already sorted, we can do better.

### Key Observation
Sorted order gives every index a yes/no property:

```text
nums[i] >= target
```

Because `nums` is sorted, this property is monotonic:

```text
False, False, False, True, True, True
```

There cannot be a `False` after a `True`. Once an element is greater than or equal to `target`, all elements to its right are also greater than or equal to `target`.

Therefore, the problem is a boundary search: find the first index where `nums[i] >= target` becomes true.

That is also known as the lower bound of `target`.

### Search Invariant
A clean way to binary search for an insertion position is to use a half-open interval:

```text
[left, right)
```

This means the active search space includes `left` but excludes `right`.

We start with:

```text
left = 0
right = len(nums)
```

The answer is always somewhere in the range `[left, right]` as an insertion boundary, and the loop searches candidate indices inside `[left, right)`.

The invariant is:

- Every index before `left` is known to contain a value less than `target`.
- Every index at or after `right` is known to be a valid insertion side for `target` because it is not needed as a smaller candidate anymore.
- The first index whose value is greater than or equal to `target`, if it has not been found exactly, is still represented by the boundary between `left` and `right`.

More practically:

```text
answer is never to the left of left
answer is never to the right of right
```

When the interval becomes empty (`left == right`), the boundary has been isolated, and that index is the answer.

### Detailed Algorithm
At each step, inspect the middle index:

```text
mid = (left + right) // 2
```

There are two possible comparison outcomes.

#### Case 1: `nums[mid] < target`
If the middle value is smaller than `target`, then `mid` cannot be the answer. Neither can any index to the left of `mid`, because those values are less than or equal to `nums[mid]`, and therefore also smaller than `target`.

So the insertion position must be strictly after `mid`:

```text
left = mid + 1
```

#### Case 2: `nums[mid] >= target`
If the middle value is greater than or equal to `target`, then `mid` is a possible answer. But it may not be the first such index. There might be an earlier value that is also greater than or equal to `target`.

So we keep `mid` in the search space and discard only the part after it:

```text
right = mid
```

This is the detail that makes the algorithm return the insertion position, not just any matching index.

### Pseudocode

```text
searchInsert(nums, target):
    left = 0
    right = length of nums

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

Equivalent Python implementation:

```python
def searchInsert(nums, target):
    left = 0
    right = len(nums)

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

### Detailed Example Walkthrough

Use `nums = [1, 3, 5, 6]` and `target = 2`.

The correct answer is `1`, because inserting `2` at index `1` gives:

```text
[1, 2, 3, 5, 6]
```

Start:

```text
left = 0, right = 4
search space = indices [0, 1, 2, 3]
```

First iteration:

```text
mid = (0 + 4) // 2 = 2
nums[mid] = nums[2] = 5
```

Since `5 >= 2`, index `2` could be an insertion position, but it might be too far right. Search the left half while keeping `2` as a possible upper boundary:

```text
right = mid = 2
```

Now:

```text
left = 0, right = 2
search space = indices [0, 1]
```

Second iteration:

```text
mid = (0 + 2) // 2 = 1
nums[mid] = nums[1] = 3
```

Since `3 >= 2`, index `1` could be the answer. Again, search left while keeping `1`:

```text
right = mid = 1
```

Now:

```text
left = 0, right = 1
search space = index [0]
```

Third iteration:

```text
mid = (0 + 1) // 2 = 0
nums[mid] = nums[0] = 1
```

Since `1 < 2`, index `0` and everything before it are too small. The answer must be after index `0`:

```text
left = mid + 1 = 1
```

Now:

```text
left = 1, right = 1
```

The interval is empty, so return `left`, which is `1`.

### Walkthrough For The Edge Positions

If `target = 5`:

```text
nums = [1, 3, 5, 6]
answer = 2
```

The binary search still finds the first index where `nums[i] >= 5`. Since `nums[2]` is exactly `5`, the insertion position is the existing index `2`.

If `target = 7`:

```text
nums = [1, 3, 5, 6]
answer = 4
```

Every element is less than `7`, so `left` keeps moving right until it equals `len(nums)`. Returning `4` means “insert after the last element.”

If `target = 0`:

```text
nums = [1, 3, 5, 6]
answer = 0
```

The first element is already greater than or equal to `0`, so the boundary is at the beginning.

### Correctness

We prove that the algorithm returns the correct insertion index.

Let the desired answer be the first index `ans` such that `nums[ans] >= target`, or `len(nums)` if no such index exists.

#### The invariant is preserved
At the start, `left = 0` and `right = len(nums)`, so `ans` is inside the possible boundary range.

During each loop:

- If `nums[mid] < target`, then every index `i <= mid` has `nums[i] < target` because the array is sorted. None of those indices can be `ans`, so setting `left = mid + 1` does not discard the answer.
- If `nums[mid] >= target`, then `mid` may be `ans`, and any later index cannot be the first valid index if `mid` itself is valid. Setting `right = mid` keeps `mid` and all earlier possible answers.

Thus every update preserves the fact that the true insertion boundary remains between `left` and `right`.

#### The loop terminates
Each iteration strictly shrinks the half-open interval `[left, right)`:

- `left = mid + 1` removes at least `mid`.
- `right = mid` removes everything from `mid + 1` onward, and because `mid < right`, the interval gets smaller.

Since the interval length is a nonnegative integer, the loop must eventually stop.

#### The returned index is correct
The loop stops when `left == right`. At that point, all indices before `left` are known to be less than `target`, and `left` is the first position not proven too small. Therefore `left` is exactly the first index whose value is greater than or equal to `target`, or `len(nums)` if no such index exists.

That is precisely the required search insert position.

### Complexity

- Time: `O(log n)` because each comparison discards about half of the remaining search interval.
- Space: `O(1)` because the algorithm stores only a few integer variables.

### Common Pitfalls

- Returning immediately when `nums[mid] == target` in a lower-bound template. It works for this problem because values are distinct, but continuing with `right = mid` is the more principled boundary-search version and also works when duplicates exist.
- Using `right = len(nums) - 1` together with `while left < right` without adjusting the return logic. That changes the interval from half-open to inclusive and requires different updates.
- Writing `right = mid - 1` in this half-open version. If `nums[mid] >= target`, `mid` itself may be the answer, so removing it can skip the correct insertion position.
- Forgetting that returning `len(nums)` is valid. If `target` is larger than every element, the insertion position is one past the last index.
- Thinking of the task as “find equality.” The real task is “find the first value not less than `target`.”

### First-Principles Summary
The sorted array divides naturally around `target`: numbers smaller than `target` belong on the left, and numbers greater than or equal to `target` belong on the right. The desired index is the boundary between those two groups.

Binary search is appropriate because one comparison at `mid` tells us which side of the boundary `mid` is on. If `nums[mid]` is too small, the boundary is to the right. Otherwise, `mid` is at or to the right of the boundary, so we keep it and search left.

The algorithm is small because it directly models that boundary. `left` advances past confirmed-too-small values, `right` moves down to the earliest still-possible valid value, and when they meet, the boundary has been found.

## Implementation
See `solutions/binary_search/p035_search_insert_position.py`.

## Tests
See `tests/binary_search/test_p035_search_insert_position.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 3, 5, 6], 'target': 5}`
- Output: `2`

### Example 2
- Input: `{'nums': [1, 3, 5, 6], 'target': 2}`
- Output: `1`

### Example 3
- Input: `{'nums': [1, 3, 5, 6], 'target': 7}`
- Output: `4`

## Follow-up Practice
- Trace why returning `left` works when `target` is smaller than every element.
- Trace why returning `left` works when `target` is larger than every element.
- Rewrite the invariant using an inclusive `[left, right]` interval and compare how the updates change.
