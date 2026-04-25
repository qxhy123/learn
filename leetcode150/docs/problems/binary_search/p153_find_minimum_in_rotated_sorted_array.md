# 153. Find Minimum in Rotated Sorted Array

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `nums` with three important properties:

```text
1. It was originally sorted in strictly increasing order.
2. It may have been rotated some number of times.
3. It contains no duplicate values.
```

A sorted array such as:

```text
[0, 1, 2, 4, 5, 6, 7]
```

can be rotated by moving a prefix to the end:

```text
[4, 5, 6, 7, 0, 1, 2]
```

The task is to return the smallest value in the rotated array.

For the example above, the answer is:

```text
0
```

because `0` is the smallest element.

The problem is not asking for the rotation count, and it is not asking us to restore the original sorted array. It only asks for the value at the rotation pivot: the place where the array drops from a larger value to a smaller value.

In a rotated sorted array, the minimum is special because it is the first element of the lower sorted part.

For example:

```text
[4, 5, 6, 7, 0, 1, 2]
             ^
          minimum
```

If the array was not rotated, the minimum is simply the first element:

```text
[11, 13, 15, 17]
 ^
minimum
```

So the real problem is:

> Find the boundary between the larger sorted part and the smaller sorted part, or recognize that no such boundary exists.

---

### 2. Start From the Brute Force Idea

The simplest approach is to scan the whole array and keep the smallest value seen:

```python
answer = nums[0]

for value in nums:
    answer = min(answer, value)

return answer
```

This is correct because every element is checked.

The time complexity is:

```text
O(n)
```

The space complexity is:

```text
O(1)
```

For many problems, this would be good enough. But this problem gives us extra structure: the array is not arbitrary. It is sorted, then rotated.

That structure should let us do better than checking every element.

The goal is to use the sortedness to discard half of the remaining search space at each step, giving:

```text
O(log n)
```

---

### 3. What Rotation Does to a Sorted Array

A normally sorted array has one increasing run:

```text
[0, 1, 2, 4, 5, 6, 7]
```

After rotation, it has two increasing runs:

```text
[4, 5, 6, 7, 0, 1, 2]
 |--------|  |-----|
 high run   low run
```

Every value in the left run is larger than every value in the right run:

```text
4, 5, 6, 7 are all greater than 0, 1, 2
```

The minimum is the first value of the right run:

```text
[4, 5, 6, 7, 0, 1, 2]
             ^
```

If the array is not rotated, there is only one run:

```text
[0, 1, 2, 4, 5, 6, 7]
 ^
```

The minimum is still the first element.

So we can think of the array as either:

```text
case 1: one sorted run
case 2: two sorted runs, where the second run starts at the minimum
```

Binary search works because a comparison can tell us which side of `mid` still contains the start of the low run.

---

### 4. The Key Observation: Compare `nums[mid]` With `nums[right]`

At any point, suppose the minimum is somewhere inside this inclusive range:

```text
nums[left:right + 1]
```

We choose:

```text
mid = (left + right) // 2
```

The most useful comparison is:

```text
nums[mid] > nums[right]
```

Why compare to the right boundary?

Because `nums[right]` belongs to the tail of the current search range. In a rotated sorted range with no duplicates, this comparison tells us whether `mid` is in the high run or the low run.

#### Case A: `nums[mid] > nums[right]`

Example:

```text
[4, 5, 6, 7, 0, 1, 2]
 left     mid         right
 nums[mid] = 7
 nums[right] = 2
```

Since `7 > 2`, `mid` is in the left high run.

The drop to the minimum must happen after `mid`:

```text
[4, 5, 6, 7, 0, 1, 2]
          mid  ^
             minimum is right of mid
```

So `mid` itself cannot be the minimum, and everything to the left of `mid` cannot contain the minimum either.

We can safely do:

```python
left = mid + 1
```

#### Case B: `nums[mid] < nums[right]`

Example:

```text
[4, 5, 6, 7, 0, 1, 2]
             mid      right
 nums[mid] = 0
 nums[right] = 2
```

Since `0 < 2`, `mid` is in a sorted increasing part that reaches the right boundary.

That means the minimum is not strictly to the right of `mid`, because all values from `mid` through `right` are sorted increasing and `nums[mid]` is the smallest value in that suffix.

The minimum could be exactly `mid`:

```text
[4, 5, 6, 7, 0, 1, 2]
             ^
```

Or it could be even earlier if the current range is already sorted:

```text
[0, 1, 2, 4, 5, 6, 7]
 left     mid         right
 ^
minimum is left of mid
```

So we must keep `mid` in the search range:

```python
right = mid
```

Not `right = mid - 1`, because `mid` itself may be the answer.

Because the problem states all values are unique, there is no third ambiguous case where `nums[mid] == nums[right]` unless `mid == right`, which cannot occur while `left < right` with the usual midpoint calculation.

---

### 5. The Search Invariant

The cleanest invariant is:

```text
At the start of every loop iteration, the minimum value is inside nums[left:right + 1].
```

The interval is inclusive on both ends.

We start with:

```text
left = 0
right = len(nums) - 1
```

So the invariant is true because the minimum is somewhere in the whole array.

Each loop step preserves the invariant.

If:

```text
nums[mid] > nums[right]
```

then the minimum is strictly to the right of `mid`, so after:

```python
left = mid + 1
```

it remains inside the new interval.

If:

```text
nums[mid] < nums[right]
```

then the minimum is at `mid` or to the left of `mid`, so after:

```python
right = mid
```

it remains inside the new interval.

The loop stops when:

```text
left == right
```

At that moment, the interval contains exactly one element. Since the invariant says the minimum is inside the interval, that one element must be the minimum.

So we return:

```python
nums[left]
```

---

### 6. Algorithm

1. Initialize the search range:

```python
left = 0
right = len(nums) - 1
```

2. While the range has more than one candidate:

```python
while left < right:
```

3. Choose the middle index:

```python
mid = (left + right) // 2
```

4. If `nums[mid]` is greater than `nums[right]`, the minimum is to the right of `mid`:

```python
left = mid + 1
```

5. Otherwise, `mid` may be the minimum, so keep the left side including `mid`:

```python
right = mid
```

6. Return the only remaining candidate:

```python
return nums[left]
```

---

### 7. Pseudocode

```python
def findMin(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]
```

This is also the intended Python implementation shape for this repository's `Solution.findMin` method.

---

### 8. Detailed Walkthrough: `[3, 4, 5, 1, 2]`

Input:

```text
nums = [3, 4, 5, 1, 2]
```

Start:

```text
left = 0, right = 4
nums[left:right + 1] = [3, 4, 5, 1, 2]
```

The minimum is somewhere in this range.

#### Iteration 1

Compute:

```text
mid = (0 + 4) // 2 = 2
nums[mid] = 5
nums[right] = 2
```

Compare:

```text
5 > 2
```

This means `mid` is in the high run:

```text
[3, 4, 5, 1, 2]
       ^     ^
      mid  minimum is after mid
```

The minimum must be to the right of `mid`, so update:

```text
left = mid + 1 = 3
right = 4
```

Remaining range:

```text
[1, 2]
```

#### Iteration 2

Compute:

```text
mid = (3 + 4) // 2 = 3
nums[mid] = 1
nums[right] = 2
```

Compare:

```text
1 < 2
```

This means the suffix from `mid` to `right` is sorted, and `mid` may be the minimum.

Update:

```text
right = mid = 3
```

Now:

```text
left = 3, right = 3
```

The range has collapsed to one element:

```text
nums[3] = 1
```

Return:

```text
1
```

---

### 9. Detailed Walkthrough: `[4, 5, 6, 7, 0, 1, 2]`

Start:

```text
left = 0, right = 6
nums = [4, 5, 6, 7, 0, 1, 2]
```

#### Iteration 1

```text
mid = 3
nums[mid] = 7
nums[right] = 2
```

Since:

```text
7 > 2
```

`mid` is in the high run, so the minimum is to the right:

```text
left = 4
right = 6
```

Remaining range:

```text
[0, 1, 2]
```

#### Iteration 2

```text
mid = 5
nums[mid] = 1
nums[right] = 2
```

Since:

```text
1 < 2
```

`mid` is in the low sorted run, and the minimum is at `mid` or to its left:

```text
left = 4
right = 5
```

Remaining range:

```text
[0, 1]
```

#### Iteration 3

```text
mid = 4
nums[mid] = 0
nums[right] = 1
```

Since:

```text
0 < 1
```

keep `mid`:

```text
left = 4
right = 4
```

Return:

```text
nums[4] = 0
```

---

### 10. What Happens When the Array Is Not Rotated?

Consider:

```text
[11, 13, 15, 17]
```

Start:

```text
left = 0, right = 3
```

#### Iteration 1

```text
mid = 1
nums[mid] = 13
nums[right] = 17
```

Since:

```text
13 < 17
```

`mid` and everything to its right form a sorted suffix. The minimum is not to the right of `mid`, so:

```text
right = mid = 1
```

#### Iteration 2

```text
left = 0, right = 1
mid = 0
nums[mid] = 11
nums[right] = 13
```

Since:

```text
11 < 13
```

keep `mid`:

```text
right = 0
```

Now:

```text
left = right = 0
```

Return:

```text
11
```

The same logic handles the unrotated case naturally. No special case is needed.

---

### 11. Why the Algorithm Is Correct

We prove correctness using the invariant:

```text
At the start of each loop iteration, the minimum element is inside nums[left:right + 1].
```

#### Initialization

At the beginning:

```text
left = 0
right = len(nums) - 1
```

The range is the entire array, so it certainly contains the minimum.

Therefore, the invariant is true before the first loop iteration.

#### Maintenance

Assume the invariant is true at the start of an iteration.

Let:

```text
mid = (left + right) // 2
```

There are two cases.

If:

```text
nums[mid] > nums[right]
```

then `mid` is in the larger left run of the rotated range. Since `nums[right]` is smaller than `nums[mid]`, the rotation drop must occur somewhere after `mid`. Therefore, the minimum is strictly in:

```text
nums[mid + 1:right + 1]
```

Updating:

```python
left = mid + 1
```

keeps the minimum inside the new range.

If:

```text
nums[mid] < nums[right]
```

then the segment from `mid` to `right` is sorted in increasing order relative to the current range. In that segment, `nums[mid]` is the smallest value. Therefore, the minimum cannot be strictly to the right of `mid`; it is either at `mid` or somewhere to the left.

Updating:

```python
right = mid
```

keeps the minimum inside the new range.

In both cases, the invariant remains true.

#### Termination

The loop stops when:

```text
left == right
```

The invariant says the minimum is inside `nums[left:right + 1]`.

But when `left == right`, that range contains exactly one element:

```text
nums[left]
```

Therefore, `nums[left]` must be the minimum, and the algorithm returns the correct value.

---

### 12. Complexity

Each iteration discards about half of the remaining candidates.

So the number of iterations is logarithmic:

```text
O(log n)
```

The algorithm only stores a few integer indices:

```text
left, right, mid
```

So the space complexity is:

```text
O(1)
```

Final complexity:

```text
Time:  O(log n)
Space: O(1)
```

---

### 13. Common Pitfalls

#### Pitfall 1: Returning `nums[0]` when the array looks sorted locally

It is tempting to check whether `nums[left] < nums[right]` and immediately return `nums[left]`.

That optimization can work if done carefully, but it is not necessary. The simpler binary search already handles the unrotated case.

Avoid adding special cases unless they make the code clearer.

#### Pitfall 2: Using `right = mid - 1`

When:

```text
nums[mid] < nums[right]
```

`mid` itself may be the minimum.

Example:

```text
[3, 4, 5, 1, 2]
          ^
         mid after the first update
```

If you use:

```python
right = mid - 1
```

you may discard the answer.

The correct update is:

```python
right = mid
```

#### Pitfall 3: Using `left = mid`

When:

```text
nums[mid] > nums[right]
```

`mid` cannot be the minimum because `nums[right]` is smaller.

So keeping `mid` is unnecessary and can cause an infinite loop for two-element ranges.

Use:

```python
left = mid + 1
```

#### Pitfall 4: Mixing inclusive and half-open intervals

This explanation uses an inclusive interval:

```text
[left, right]
```

That is why the loop condition is:

```python
while left < right:
```

and the answer is:

```python
nums[left]
```

If you switch to a half-open interval like `[left, right)`, the update rules and loop condition must change too. Do not mix the two styles.

#### Pitfall 5: Forgetting the no-duplicates condition

This LeetCode problem has no duplicate values.

That is why comparing `nums[mid]` and `nums[right]` always gives a useful direction while `left < right`.

If duplicates were allowed, a case like this could occur:

```text
[2, 2, 2, 0, 1, 2]
```

Then `nums[mid] == nums[right]` may be ambiguous, and a different version of the algorithm is needed. That is a separate problem.

---

### 14. First-Principles Summary

A rotated sorted array is made of one or two increasing runs.

The minimum is either:

```text
the first element, if the array was not rotated
```

or:

```text
the first element of the lower run, if the array was rotated
```

Binary search works because `nums[mid]` compared with `nums[right]` tells us which side can still contain that boundary.

If:

```text
nums[mid] > nums[right]
```

then `mid` is in the high run, so the minimum must be to the right.

If:

```text
nums[mid] < nums[right]
```

then `mid` is in the low sorted suffix, so the minimum is at `mid` or to the left.

The invariant is simple:

```text
The minimum is always inside nums[left:right + 1].
```

Every update keeps that invariant true while shrinking the interval. When only one candidate remains, it must be the minimum.

## Implementation
See `solutions/binary_search/p153_find_minimum_in_rotated_sorted_array.py`.

## Tests
See `tests/binary_search/test_p153_find_minimum_in_rotated_sorted_array.py`.

## Examples

### Example 1
- Input: `{'nums': [3, 4, 5, 1, 2]}`
- Output: `1`

### Example 2
- Input: `{'nums': [4, 5, 6, 7, 0, 1, 2]}`
- Output: `0`

### Example 3
- Input: `{'nums': [11, 13, 15, 17]}`
- Output: `11`

## Follow-up Practice
- Trace the invariant on arrays of length `1`, `2`, and `3`.
- Explain why `right = mid` is correct but `right = mid - 1` is not.
- Compare this problem with the duplicate-allowed variant and identify where equality becomes ambiguous.
