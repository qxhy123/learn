# 88. Merge Sorted Array

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/merge-sorted-array/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: in-place, reverse-merge, two-pointers

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two sorted integer arrays:

```text
nums1 = first sorted array, followed by extra empty slots
nums2 = second sorted array
```

The values that actually matter are:

```text
nums1[0:m]
nums2[0:n]
```

The last `n` positions of `nums1` are not meaningful input values. They are reserved space where the final merged result can be written.

For example:

```text
nums1 = [1, 2, 3, 0, 0, 0], m = 3
nums2 = [2, 5, 6],          n = 3
```

Only this prefix of `nums1` is real input:

```text
[1, 2, 3]
```

The trailing zeroes are just capacity:

```text
[_, _, _]
```

The task is to mutate `nums1` so that all `m + n` real values appear in sorted order:

```text
[1, 2, 2, 3, 5, 6]
```

The important constraints are:

```text
nums1 already has enough space
the merge must be written into nums1
both input portions are already sorted
```

So the problem is not "how do we sort two arrays?" It is:

> How do we combine two already-sorted runs into the storage of the first array without destroying values from `nums1` that we have not used yet?

---

### 2. Start From the Baseline

The simplest correct idea is to ignore the in-place challenge:

1. Copy the first `m` values of `nums1`.
2. Combine that copy with `nums2`.
3. Sort or merge into a new temporary array.
4. Copy the result back into `nums1`.

For example:

```python
tmp = nums1[:m] + nums2
tmp.sort()
nums1[:] = tmp
```

This is easy to reason about because no input value is overwritten before being copied somewhere safe.

But it misses the point of the reserved space in `nums1`.

If we sort the combined list, the time cost is:

```text
O((m + n) log(m + n))
```

If we do a normal forward merge into a temporary array, the time improves to:

```text
O(m + n)
```

but the extra space is:

```text
O(m + n)
```

The deeper question is:

> Can we get the linear-time merge while using the empty slots already available in `nums1` instead of allocating another array?

Yes, but only if we write in an order that does not overwrite unread values.

---

### 3. Why a Forward In-Place Merge Is Dangerous

A normal merge usually compares the smallest remaining values and writes the smaller one first.

That suggests pointers like:

```text
i = start of nums1's real values
j = start of nums2
write = start of nums1
```

But that is unsafe because `write` starts inside the same part of `nums1` that still contains unread input.

Consider:

```text
nums1 = [4, 5, 6, 0, 0, 0], m = 3
nums2 = [1, 2, 3],          n = 3
```

The smallest value is `1`, so a forward merge would want to write it at `nums1[0]`.

But `nums1[0]` currently holds `4`, which has not been merged yet:

```text
write 1 at index 0

nums1 becomes [1, 5, 6, 0, 0, 0]
lost value: 4
```

Once `4` is overwritten, the algorithm cannot recover the correct answer.

This tells us the central hazard:

> The front of `nums1` contains unread input, so writing from the front can destroy information.

---

### 4. The Key Observation: The Back Is Empty

The extra storage in `nums1` is at the end.

That changes the natural direction of the merge.

Instead of repeatedly placing the smallest remaining value at the front, place the largest remaining value at the back.

Why is this safe?

The largest remaining value among two sorted arrays must be one of:

```text
nums1[m - 1]
nums2[n - 1]
```

because each input portion is sorted.

And the final position where the largest remaining value belongs is:

```text
nums1[m + n - 1]
```

That position is either part of the reserved space or a position whose original value has already been moved/consumed by the time we write there.

So the safe direction is:

```text
read from the ends
write from the end
move backward
```

This avoids the overwrite problem because the unread `nums1` values are at indices `0 ... i`, while the write pointer starts at the far right and moves left.

---

### 5. State and Invariant

Use three pointers:

```text
i = index of the largest unmerged value in nums1[0:m]
j = index of the largest unmerged value in nums2[0:n]
k = index where the next largest value should be written in nums1
```

Initially:

```text
i = m - 1
j = n - 1
k = m + n - 1
```

Maintain this invariant:

```text
nums1[k + 1 : m + n] already contains the largest merged values
from nums1[0 : i + 1] and nums2[0 : j + 1], in correct sorted order.
```

The unmerged values are still exactly:

```text
nums1[0 : i + 1]
nums2[0 : j + 1]
```

At each step, the largest unmerged value is either:

```text
nums1[i]
```

or:

```text
nums2[j]
```

So we compare those two values, write the larger one to `nums1[k]`, and move the pointer that supplied it.

This is the whole algorithm.

---

### 6. Detailed Algorithm

1. Set `i = m - 1`, the last real element in `nums1`.
2. Set `j = n - 1`, the last element in `nums2`.
3. Set `k = m + n - 1`, the last position in `nums1`.
4. While both arrays still have unmerged values:
   - Compare `nums1[i]` and `nums2[j]`.
   - Write the larger value into `nums1[k]`.
   - Move the pointer that supplied that value backward.
   - Move `k` backward.
5. If any values remain in `nums2`, copy them into the front of `nums1`.
6. If values remain in `nums1`, do nothing. They are already in their correct positions.

That last point is subtle and important.

If `nums2` is exhausted first, then the remaining values from `nums1` are already sitting in the front of `nums1`. Since they are the smallest remaining values and already sorted, they do not need to be moved.

If `nums1` is exhausted first, then remaining values from `nums2` must be copied, because they are not already in `nums1`.

---

### 7. Pseudocode

```text
i = m - 1
j = n - 1
k = m + n - 1

while i >= 0 and j >= 0:
    if nums1[i] > nums2[j]:
        nums1[k] = nums1[i]
        i -= 1
    else:
        nums1[k] = nums2[j]
        j -= 1
    k -= 1

while j >= 0:
    nums1[k] = nums2[j]
    j -= 1
    k -= 1
```

---

### 8. Python Implementation Shape

LeetCode expects the method to modify `nums1` in place. The return value is ignored.

```python
from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i = m - 1
        j = n - 1
        k = m + n - 1

        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
```

Notice that there is no loop to copy leftover `nums1` values. They are already where they belong.

---

### 9. Example Walkthrough

Use the first official example:

```text
nums1 = [1, 2, 3, 0, 0, 0], m = 3
nums2 = [2, 5, 6],          n = 3
```

Initial pointers:

```text
i = 2  -> nums1[i] = 3
j = 2  -> nums2[j] = 6
k = 5
```

#### Step 1

Compare:

```text
nums1[i] = 3
nums2[j] = 6
```

The larger value is `6`, so write it at the final position:

```text
nums1 = [1, 2, 3, 0, 0, 6]
```

Move `j` and `k`:

```text
i = 2, j = 1, k = 4
```

The suffix `[6]` is now finalized.

#### Step 2

Compare:

```text
nums1[i] = 3
nums2[j] = 5
```

Write `5`:

```text
nums1 = [1, 2, 3, 0, 5, 6]
```

Move `j` and `k`:

```text
i = 2, j = 0, k = 3
```

The suffix `[5, 6]` is finalized.

#### Step 3

Compare:

```text
nums1[i] = 3
nums2[j] = 2
```

Write `3`:

```text
nums1 = [1, 2, 3, 3, 5, 6]
```

Move `i` and `k`:

```text
i = 1, j = 0, k = 2
```

The suffix `[3, 5, 6]` is finalized.

It may look strange that `3` appears twice temporarily. That is fine. The old `3` at index `2` was copied backward into its final position, and index `2` is still available to be overwritten later.

#### Step 4

Compare:

```text
nums1[i] = 2
nums2[j] = 2
```

The values are equal. Either one can be placed next without breaking sorted order. This implementation chooses `nums2[j]` when values are equal:

```text
nums1 = [1, 2, 2, 3, 5, 6]
```

Move `j` and `k`:

```text
i = 1, j = -1, k = 1
```

Now `nums2` is exhausted.

The remaining values from `nums1` are:

```text
nums1[0:2] = [1, 2]
```

They are already in place, so the algorithm stops.

Final result:

```text
[1, 2, 2, 3, 5, 6]
```

---

### 10. Why The Algorithm Is Correct

We prove correctness using the invariant from above.

At any moment:

```text
nums1[k + 1 : m + n]
```

contains the largest values that have already been chosen, in their final sorted positions.

#### Initialization

Before the loop starts:

```text
k = m + n - 1
```

So the finalized suffix is:

```text
nums1[m + n : m + n]
```

That suffix is empty. An empty suffix is trivially sorted and contains exactly the zero largest chosen values.

The invariant holds.

#### Maintenance

Assume the invariant holds before one loop iteration.

The unmerged values are:

```text
nums1[0 : i + 1]
nums2[0 : j + 1]
```

Because both portions are sorted, the largest unmerged value in `nums1` is `nums1[i]`, and the largest unmerged value in `nums2` is `nums2[j]`.

Therefore the largest unmerged value overall is:

```text
max(nums1[i], nums2[j])
```

The next open final position is `nums1[k]`, immediately before the already-finalized suffix. The algorithm writes exactly that largest unmerged value into `nums1[k]`.

After writing, it moves the source pointer backward and also moves `k` backward. The finalized suffix is now one element longer, and it contains the correct largest values in sorted order.

The invariant is preserved.

#### Termination

The main loop stops when at least one input portion is exhausted.

If `nums2` is exhausted, all remaining values are from the beginning of `nums1`. They are already sorted and already occupy the remaining front positions, so the final array is correct.

If `nums1` is exhausted, the remaining values of `nums2` must be the smallest remaining values. The second loop copies them into the remaining front positions of `nums1`, preserving their sorted order because it also copies from right to left.

In both cases, every value from the two input portions appears exactly once in `nums1`, and the array is sorted.

Therefore the algorithm produces the required merged array in place.

---

### 11. Complexity

Each real input value is considered at most once.

The pointers only move backward:

```text
i moves from m - 1 down to -1
j moves from n - 1 down to -1
k moves from m + n - 1 down to 0
```

So the time complexity is:

```text
O(m + n)
```

The algorithm uses only three integer pointers and writes into the existing `nums1` storage.

So the extra space complexity is:

```text
O(1)
```

---

### 12. Common Pitfalls

- Writing from the front of `nums1` can overwrite unmerged values from the original `nums1` prefix.
- Treating the trailing zeroes in `nums1` as real values is incorrect; only `nums1[0:m]` is meaningful input.
- Forgetting to copy remaining `nums2` values gives the wrong result when all original `nums1` values are larger, such as `nums1 = [4,5,6,0,0,0]` and `nums2 = [1,2,3]`.
- Copying remaining `nums1` values is unnecessary and can introduce off-by-one mistakes; they are already in place.
- Using `>=` versus `>` in the comparison does not affect sorted correctness. With equal values, either copy is valid because equal numbers are interchangeable for this problem.
- Returning a new list does not satisfy the in-place requirement. The judge checks the mutated contents of `nums1`.
- In tests outside LeetCode, remember that the conventional method returns `None`; inspect `nums1` after calling the method.

---

### 13. First-Principles Summary

The entire problem is driven by one storage fact:

```text
the front of nums1 contains unread values, but the back of nums1 contains free space
```

That means a forward merge risks destroying information, while a reverse merge uses the empty capacity exactly where it is safe to write.

Because both input portions are sorted, the largest remaining value is always visible at one of the two right ends. Put that value into the rightmost open output position, then move backward.

The invariant is:

```text
the suffix after k is already the correct final suffix
```

Once that is true, every step is forced: choose the larger right-end value, write it at `k`, and shrink the unmerged region.

This is why the solution is linear, in place, and simpler than trying to shift values forward.

## Implementation

See `solutions/array_string/p088_merge_sorted_array.py`.

## Tests

See `tests/array_string/test_p088_merge_sorted_array.py`.

## Examples

### Example 1
- Input: `{'nums1': [1, 2, 3, 0, 0, 0], 'm': 3, 'nums2': [2, 5, 6], 'n': 3}`
- Output: `[1, 2, 2, 3, 5, 6]`

### Example 2
- Input: `{'nums1': [1], 'm': 1, 'nums2': [], 'n': 0}`
- Output: `[1]`

### Example 3
- Input: `{'nums1': [0], 'm': 0, 'nums2': [1], 'n': 1}`
- Output: `[1]`

## Follow-up Practice

- Trace `i`, `j`, and `k` on an input where every `nums2` value is smaller than every `nums1` value.
- Trace an input where `n = 0`; the array should not change.
- Trace an input where `m = 0`; every value must be copied from `nums2`.
- Explain why the algorithm does not need to copy leftover `nums1` values.
