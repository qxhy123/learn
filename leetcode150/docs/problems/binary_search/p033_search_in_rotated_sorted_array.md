# 33. Search in Rotated Sorted Array

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/search-in-rotated-sorted-array/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer array `nums` with two important properties:

```text
1. It was originally sorted in strictly increasing order.
2. It may have been rotated at some pivot.
```

A sorted array such as:

```text
[0, 1, 2, 4, 5, 6, 7]
```

can be rotated into:

```text
[4, 5, 6, 7, 0, 1, 2]
```

The relative order inside each piece is still sorted, but the smallest value no longer has to be at index `0`.

Given a `target`, return its index if it exists. If it does not exist, return `-1`.

For example:

```text
nums   = [4, 5, 6, 7, 0, 1, 2]
target = 0
```

The value `0` is at index `4`, so the answer is:

```text
4
```

If:

```text
target = 3
```

then `3` is not in the array, so the answer is:

```text
-1
```

The real problem is:

> Find one value inside an almost-sorted array while still getting logarithmic search time.

The phrase "rotated sorted array" matters because the array is not fully sorted from left to right anymore, so ordinary binary search cannot be applied blindly.

---

### 2. Start From the Brute Force Baseline

The simplest approach is linear search:

```python
for i in range(len(nums)):
    if nums[i] == target:
        return i

return -1
```

This is correct because it checks every possible index.

But it costs:

```text
O(n) time
```

The problem is designed to use more structure than that. A rotated sorted array is not arbitrary. It still contains a strong ordering pattern, and that pattern should let us discard many candidates at once.

So the deeper question is:

> Even though the whole array is not globally sorted, can every binary-search step still identify a half that is safely discardable?

Yes.

---

### 3. Why Ordinary Binary Search Is Not Enough

In a normal sorted array, binary search compares `target` with `nums[mid]`.

If:

```text
target < nums[mid]
```

then the target must be on the left.

If:

```text
target > nums[mid]
```

then the target must be on the right.

That works because every value left of `mid` is smaller than `nums[mid]`, and every value right of `mid` is larger than `nums[mid]`.

A rotated sorted array breaks that global rule.

For:

```text
nums = [4, 5, 6, 7, 0, 1, 2]
```

if `mid` points to `7`, then values to the right are not larger than `7`; they are `[0, 1, 2]`.

So this reasoning is invalid:

```text
target < nums[mid]  =>  target must be left
```

For example, `target = 0` is less than `7`, but it is on the right.

We need a better local question than just "is target smaller or larger than `nums[mid]`?"

---

### 4. The Key Observation: One Half Is Always Sorted

Although the entire remaining interval may be rotated, splitting it at `mid` gives two halves:

```text
nums[left ... mid]
nums[mid ... right]
```

At least one of these halves is sorted in normal increasing order.

Why?

A rotation creates one "break" point, where a large value is followed by a small value:

```text
[4, 5, 6, 7, 0, 1, 2]
             ^ break between 7 and 0
```

When we split an interval into two halves, that single break can be in at most one half.

Therefore, the other half has no break and is sorted.

That gives us the binary-search decision we need:

```text
Find which half is sorted.
Then ask whether target lies within that sorted half's value range.
```

If the target lies inside the sorted half's range, search that half.

If it does not, discard that sorted half and search the other half.

This is the central idea of the problem.

---

### 5. The Search Invariant

Use an inclusive search interval:

```text
[left, right]
```

Maintain this invariant:

```text
If target exists in nums, then target is inside nums[left ... right].
```

At the start:

```text
left = 0
right = len(nums) - 1
```

So the invariant is true because the interval covers the entire array.

On every iteration:

1. Compute `mid`.
2. If `nums[mid] == target`, return `mid` immediately.
3. Otherwise, identify which side is sorted.
4. Keep only the side that can still contain `target`.

The update must preserve the invariant. That means we are allowed to discard a side only after proving the target cannot be there.

The interval must also strictly shrink. Otherwise, the loop could run forever.

---

### 6. How to Identify the Sorted Half

Because values are distinct in this problem, the comparison is clean.

If:

```text
nums[left] <= nums[mid]
```

then the left half is sorted:

```text
nums[left ... mid]
```

Why does `<=` work?

Because if the left boundary value is less than or equal to the middle value, there is no rotation break between `left` and `mid`. So that half is increasing.

Otherwise:

```text
nums[left] > nums[mid]
```

then the rotation break lies somewhere in the left half, which means the right half must be sorted:

```text
nums[mid ... right]
```

You can also reason with `nums[mid] <= nums[right]` for the right half. The important part is to consistently identify one sorted side and test the target against that side's value range.

---

### 7. Deciding Which Half to Keep

#### Case A: The Left Half Is Sorted

If:

```text
nums[left] <= nums[mid]
```

then:

```text
nums[left ... mid]
```

is sorted.

The target is inside this sorted half exactly when:

```text
nums[left] <= target < nums[mid]
```

The upper comparison is `< nums[mid]`, not `<= nums[mid]`, because equality with `nums[mid]` was already handled earlier.

If the target is in that range, keep the left half:

```text
right = mid - 1
```

Otherwise, the target cannot be in the sorted left half, so keep the right half:

```text
left = mid + 1
```

#### Case B: The Right Half Is Sorted

Otherwise, the right half is sorted:

```text
nums[mid ... right]
```

The target is inside this sorted half exactly when:

```text
nums[mid] < target <= nums[right]
```

Again, the lower comparison is strict because `nums[mid] == target` was already checked.

If the target is in that range, keep the right half:

```text
left = mid + 1
```

Otherwise, keep the left half:

```text
right = mid - 1
```

---

### 8. Detailed Algorithm

1. Initialize:

```text
left = 0
right = len(nums) - 1
```

2. While `left <= right`:

```text
mid = (left + right) // 2
```

3. If the middle value is the target:

```text
return mid
```

4. If the left half is sorted:

```text
nums[left] <= nums[mid]
```

then check whether `target` lies in the sorted range:

```text
nums[left] <= target < nums[mid]
```

If yes, move `right` leftward. If no, move `left` rightward.

5. Otherwise, the right half is sorted. Check whether `target` lies in the sorted range:

```text
nums[mid] < target <= nums[right]
```

If yes, move `left` rightward. If no, move `right` leftward.

6. If the loop ends, the target is not present:

```text
return -1
```

---

### 9. Pseudocode

```python
def search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted.
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        # Right half is sorted.
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

This is still binary search, but the discard decision is rotation-aware.

Instead of asking only:

```text
Is target less than nums[mid]?
```

we ask:

```text
Which half is sorted, and does target fit inside that half's value range?
```

---

### 10. Walkthrough: Target Exists

Use the first example:

```text
nums   = [4, 5, 6, 7, 0, 1, 2]
target = 0
```

Start:

```text
left = 0, right = 6
nums[left] = 4, nums[right] = 2
```

#### Iteration 1

```text
mid = (0 + 6) // 2 = 3
nums[mid] = 7
```

`7` is not the target.

Check which half is sorted:

```text
nums[left] <= nums[mid]
4 <= 7  => true
```

So the left half is sorted:

```text
[4, 5, 6, 7]
```

Does `target = 0` lie in this sorted range?

```text
nums[left] <= target < nums[mid]
4 <= 0 < 7  => false
```

So `0` cannot be in `[4, 5, 6, 7]`.

Discard the left half:

```text
left = mid + 1 = 4
```

Remaining interval:

```text
[0, 1, 2]
 ^     ^
left  right
```

#### Iteration 2

```text
left = 4, right = 6
mid = (4 + 6) // 2 = 5
nums[mid] = 1
```

`1` is not the target.

Check which half is sorted:

```text
nums[left] <= nums[mid]
0 <= 1  => true
```

So the left half is sorted:

```text
[0, 1]
```

Does `target = 0` lie in this sorted range before `mid`?

```text
nums[left] <= target < nums[mid]
0 <= 0 < 1  => true
```

So the target must be on the left side of `mid`.

Update:

```text
right = mid - 1 = 4
```

Remaining interval:

```text
[0]
 ^
left/right
```

#### Iteration 3

```text
left = 4, right = 4
mid = 4
nums[mid] = 0
```

This equals the target, so return:

```text
4
```

---

### 11. Walkthrough: Target Missing

Use the second example:

```text
nums   = [4, 5, 6, 7, 0, 1, 2]
target = 3
```

#### Iteration 1

```text
left = 0, right = 6
mid = 3
nums[mid] = 7
```

Left half is sorted because:

```text
nums[left] <= nums[mid]
4 <= 7
```

But `3` is not in the sorted range `[4, 7)`:

```text
4 <= 3 < 7  => false
```

So search right:

```text
left = 4
```

#### Iteration 2

```text
left = 4, right = 6
mid = 5
nums[mid] = 1
```

Left half is sorted because:

```text
nums[left] <= nums[mid]
0 <= 1
```

But `3` is not in `[0, 1)`:

```text
0 <= 3 < 1  => false
```

So search right:

```text
left = 6
```

#### Iteration 3

```text
left = 6, right = 6
mid = 6
nums[mid] = 2
```

Left half is sorted because it contains one element:

```text
nums[left] <= nums[mid]
2 <= 2
```

But `3` is not in `[2, 2)`:

```text
2 <= 3 < 2  => false
```

So move right again:

```text
left = 7
```

Now:

```text
left > right
```

The search interval is empty. By the invariant, if the target existed, it would have to be inside the interval. Since there is no interval left, return:

```text
-1
```

---

### 12. Correctness

We prove that the algorithm returns the correct index if `target` exists, and returns `-1` otherwise.

#### Invariant

At the start of every loop iteration:

```text
If target exists in nums, then target is inside nums[left ... right].
```

#### Initialization

Before the first iteration:

```text
left = 0
right = len(nums) - 1
```

The interval covers the whole array. Therefore, if the target exists, it is inside `nums[left ... right]`.

So the invariant holds initially.

#### Maintenance

During an iteration, the algorithm first checks `nums[mid]`.

If:

```text
nums[mid] == target
```

then returning `mid` is correct.

Otherwise, `mid` is not the answer.

Now there are two cases.

If the left half is sorted, then every value in `nums[left ... mid]` lies in increasing order between `nums[left]` and `nums[mid]`.

- If `nums[left] <= target < nums[mid]`, the target can only be in that sorted left half, so setting `right = mid - 1` keeps all possible target positions.
- Otherwise, the target cannot be in that sorted left half, and `mid` is already known not to be the answer, so setting `left = mid + 1` keeps all possible target positions.

If the right half is sorted, then every value in `nums[mid ... right]` lies in increasing order between `nums[mid]` and `nums[right]`.

- If `nums[mid] < target <= nums[right]`, the target can only be in that sorted right half, so setting `left = mid + 1` keeps all possible target positions.
- Otherwise, the target cannot be in that sorted right half, and `mid` is already known not to be the answer, so setting `right = mid - 1` keeps all possible target positions.

In every case, the algorithm discards only indices that cannot contain the target.

Therefore, the invariant is preserved.

#### Termination

Each iteration either returns immediately or removes `mid` and at least one side of the interval from consideration.

So the interval strictly shrinks.

Eventually either the target is found, or:

```text
left > right
```

At that point the interval is empty. By the invariant, if the target existed, it would have to be inside the empty interval, which is impossible.

Therefore, returning `-1` is correct.

---

### 13. Complexity

Each iteration discards about half of the remaining search interval.

So the number of iterations is logarithmic:

```text
O(log n)
```

The algorithm stores only a few integer indices:

```text
left, right, mid
```

So the extra space usage is:

```text
O(1)
```

Final complexity:

```text
Time:  O(log n)
Space: O(1)
```

---

### 14. Common Pitfalls

#### Pitfall 1: Using Ordinary Binary Search Directly

This is wrong:

```python
if target < nums[mid]:
    right = mid - 1
else:
    left = mid + 1
```

In a rotated array, smaller values may appear to the right of a larger middle value.

You must first identify the sorted half.

#### Pitfall 2: Forgetting to Check `nums[mid]` First

The range tests often use strict comparisons around `nums[mid]`:

```text
nums[left] <= target < nums[mid]
nums[mid] < target <= nums[right]
```

These assume `nums[mid]` has already been checked.

If you skip the equality check, you can accidentally discard the answer.

#### Pitfall 3: Mixing Boundary Styles

This explanation uses an inclusive interval:

```text
[left, right]
```

So the loop condition is:

```text
left <= right
```

and the updates are:

```text
left = mid + 1
right = mid - 1
```

Do not mix this with half-open interval updates unless you rewrite the invariant.

#### Pitfall 4: Mishandling Single-Element Intervals

When `left == right`, the interval has one element.

The check:

```text
nums[left] <= nums[mid]
```

is true because `left == mid`.

That is fine. The equality check for `nums[mid] == target` happens first. If it is not the target, the algorithm will move past it and terminate.

#### Pitfall 5: Assuming This Version Handles Duplicates

LeetCode 33 uses distinct values.

If duplicates are allowed, comparisons such as:

```text
nums[left] <= nums[mid]
```

may not clearly identify the sorted half. That is a different problem, usually handled by carefully shrinking ambiguous boundaries.

---

### 15. First-Principles Summary

A rotated sorted array is not random. It is two sorted pieces joined together.

The rotation creates at most one break in increasing order.

When we split the current search interval at `mid`, that break can only be on one side, so at least one half is sorted.

That sorted half gives us a reliable value range test:

```text
If target belongs to the sorted half's range, keep it.
Otherwise, discard it.
```

The invariant is:

```text
If the target exists, it is always inside the current [left, right] interval.
```

Every update preserves that invariant and strictly shrinks the interval.

So the algorithm is binary search with a stronger question at each step:

```text
Which side is sorted, and can the target be inside that side?
```

That is why the solution achieves `O(log n)` time even though the array is rotated.

## Implementation
See `solutions/binary_search/p033_search_in_rotated_sorted_array.py`.

## Tests
See `tests/binary_search/test_p033_search_in_rotated_sorted_array.py`.

## Examples

### Example 1
- Input: `{'nums': [4, 5, 6, 7, 0, 1, 2], 'target': 0}`
- Output: `4`

### Example 2
- Input: `{'nums': [4, 5, 6, 7, 0, 1, 2], 'target': 3}`
- Output: `-1`

### Example 3
- Input: `{'nums': [1], 'target': 0}`
- Output: `-1`

## Follow-up Practice
- Trace the algorithm on arrays rotated by `0`, `1`, and `n - 1` positions.
- Explain why at least one half must be sorted after every split.
- Rewrite the invariant before coding the loop.
- Compare the inclusive `[left, right]` version with a half-open `[left, right)` version.
- Think through how duplicates would make sorted-half detection ambiguous.
