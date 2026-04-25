# 26. Remove Duplicates from Sorted Array

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: in-place, slow-fast-pointers

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer array `nums` sorted in non-decreasing order.

That means:

```text
nums[i] <= nums[i + 1]
```

The task is to remove duplicate values in-place so that each distinct value appears exactly once in the front part of the same array.

Two details matter more than anything else:

```text
1. The array is sorted.
2. The change must be done in-place.
```

The function does not need to physically shrink the Python list. Instead, it returns an integer `k`.

After the function returns:

```text
nums[0:k] must contain the unique values in sorted order
anything after index k - 1 does not matter
```

For example:

```text
nums = [1, 1, 2]
```

The correct result is:

```text
k = 2
nums[0:2] = [1, 2]
```

The remaining cell may still contain an old value:

```text
[1, 2, 2]
```

LeetCode displays that unused suffix as:

```text
[1, 2, _]
```

So the real problem is:

> Rewrite the front of the array so it contains one representative of each distinct value, preserving order, and return the number of representatives written.

---

### 2. Start From a Baseline Idea

If we ignore the in-place requirement, the simplest solution is to build a new array:

```python
unique = []

for x in nums:
    if not unique or unique[-1] != x:
        unique.append(x)

return len(unique)
```

This works because the input is sorted. Equal values appear in one contiguous block, so the first value of a block is the only one we need to keep.

For:

```text
[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
```

the blocks are:

```text
[0, 0] [1, 1, 1] [2, 2] [3, 3] [4]
```

Taking the first value from each block gives:

```text
[0, 1, 2, 3, 4]
```

The problem is that this baseline uses extra space proportional to the number of unique values. The challenge asks us to place those values back into `nums` itself.

So the question becomes:

> Can we use the original array as the output array while we scan it?

Yes. We only need to know where the next kept value should be written.

---

### 3. The Key Observation: Sorted Values Form Runs

Because the array is sorted, duplicates are adjacent.

If we scan from left to right, a new unique value starts exactly when the current value differs from the last unique value we kept:

```text
current value != last kept value
```

There is no need to search the whole prefix to know whether `nums[i]` appeared before.

For example:

```text
[1, 1, 1, 2, 2, 3]
          ^
```

When we reach the first `2`, we only need to compare it with the last kept value `1`.

If it differs, it cannot be a duplicate of anything earlier, because all earlier values are `<= 1` or equal to `1`. Sorted order guarantees that all copies of `1` have already ended.

This is the core reason the solution can be one pass and constant space.

---

### 4. What State Do We Need?

We need two positions:

```text
read  = index of the value currently being inspected
write = next position where a new unique value should be placed
```

The front of the array is treated as the output buffer:

```text
nums[0:write]
```

The invariant is:

```text
Before each read step, nums[0:write] contains exactly the unique values
from the already processed part of the array, in sorted order.
```

This is the whole algorithm. Everything else is just maintaining that invariant.

At each `read` index:

```text
If nums[read] is the same as the previous input value,
it is a duplicate within the same sorted run, so skip it.

If nums[read] differs from the previous input value,
it begins a new run, so copy it to nums[write] and advance write.
```

There is an equivalent comparison:

```text
nums[read] != nums[write - 1]
```

Here `nums[write - 1]` is the last unique value already kept.

---

### 5. Why Overwriting Is Safe

In-place array problems often raise an important concern:

> If we write into `nums`, could we destroy a value we still need to read later?

Here the answer is no.

The `write` pointer never moves ahead of the `read` pointer:

```text
write <= read + 1
```

Usually `write` is behind `read`, because duplicates create gaps between the compacted output and the original scan position.

When we execute:

```python
nums[write] = nums[read]
```

we write to a position at or behind the current read location. All positions before `read` have already been inspected. Positions after `read` are untouched and will still be available later.

So the original array safely doubles as the output buffer.

---

### 6. Detailed Algorithm

Handle the smallest input first:

```text
If nums is empty, return 0.
```

LeetCode's original constraints usually give at least one element, but the empty-array guard makes the logic complete and easy to reason about.

For a non-empty array:

```text
The first element is always unique within the processed prefix.
```

So initialize:

```text
write = 1
```

This means:

```text
nums[0:1] already contains the first unique value.
```

Then scan from the second element:

```text
for read from 1 to len(nums) - 1:
    if nums[read] != nums[write - 1]:
        nums[write] = nums[read]
        write += 1
```

Finally:

```text
return write
```

Why does `write` equal the answer?

Because `write` always points one position past the compacted unique prefix. If five unique values have been written, they occupy:

```text
nums[0], nums[1], nums[2], nums[3], nums[4]
```

and `write` is `5`.

---

### 7. Pseudocode

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        write = 1

        for read in range(1, len(nums)):
            if nums[read] != nums[write - 1]:
                nums[write] = nums[read]
                write += 1

        return write
```

Some versions compare against the previous raw input value instead:

```python
if nums[read] != nums[read - 1]:
    nums[write] = nums[read]
    write += 1
```

That is also correct for this problem because the input is sorted. The `nums[write - 1]` version makes the invariant especially visible: compare the candidate against the last value kept in the output prefix.

---

### 8. Walkthrough: `[1, 1, 2]`

Initial array:

```text
nums = [1, 1, 2]
```

The first value is kept automatically:

```text
write = 1
kept prefix = [1]
```

#### Read index 1

```text
nums[read] = 1
last kept  = nums[write - 1] = nums[0] = 1
```

They are equal, so this is a duplicate.

Skip it:

```text
nums = [1, 1, 2]
write = 1
kept prefix = [1]
```

#### Read index 2

```text
nums[read] = 2
last kept  = nums[write - 1] = nums[0] = 1
```

They differ, so `2` starts a new sorted run.

Write it at `write`:

```text
nums[1] = 2
write = 2
```

Now:

```text
nums = [1, 2, 2]
kept prefix = [1, 2]
```

Return:

```text
2
```

Only the first two cells matter. The final `2` is outside the returned length.

---

### 9. Walkthrough: `[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]`

Start:

```text
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
write = 1
kept prefix = [0]
```

Now scan:

```text
read = 1, nums[read] = 0
last kept = 0
duplicate, skip
write = 1
kept = [0]
```

```text
read = 2, nums[read] = 1
last kept = 0
new value, write nums[1] = 1
write = 2
kept = [0, 1]
```

```text
read = 3, nums[read] = 1
last kept = 1
duplicate, skip
write = 2
kept = [0, 1]
```

```text
read = 4, nums[read] = 1
last kept = 1
duplicate, skip
write = 2
kept = [0, 1]
```

```text
read = 5, nums[read] = 2
last kept = 1
new value, write nums[2] = 2
write = 3
kept = [0, 1, 2]
```

```text
read = 6, nums[read] = 2
last kept = 2
duplicate, skip
write = 3
kept = [0, 1, 2]
```

```text
read = 7, nums[read] = 3
last kept = 2
new value, write nums[3] = 3
write = 4
kept = [0, 1, 2, 3]
```

```text
read = 8, nums[read] = 3
last kept = 3
duplicate, skip
write = 4
kept = [0, 1, 2, 3]
```

```text
read = 9, nums[read] = 4
last kept = 3
new value, write nums[4] = 4
write = 5
kept = [0, 1, 2, 3, 4]
```

Return:

```text
5
```

The first five values of `nums` are:

```text
[0, 1, 2, 3, 4]
```

The remaining positions are irrelevant.

---

### 10. Correctness

We prove that the algorithm returns the number of unique values and places those values in `nums[0:k]` in sorted order.

#### Invariant

Before each iteration with index `read`, the prefix:

```text
nums[0:write]
```

contains exactly the distinct values from the original processed prefix:

```text
original nums[0:read]
```

in the same sorted order, with no duplicates.

#### Initialization

Before the loop starts, `read = 1` and `write = 1`.

The processed prefix is the first element:

```text
original nums[0:1]
```

It has exactly one distinct value, and `nums[0:1]` contains it. So the invariant holds.

If the array is empty, the algorithm returns `0`, which is correct because there are no unique values.

#### Maintenance

Assume the invariant is true before processing `nums[read]`.

There are two cases.

Case 1:

```text
nums[read] == nums[write - 1]
```

The current value is equal to the last unique value already kept. Since the array is sorted, this value is another copy of the current run. Skipping it does not remove any distinct value from the output prefix. The invariant remains true for the next `read`.

Case 2:

```text
nums[read] != nums[write - 1]
```

The current value differs from the last unique value. Since the array is sorted, it must be the first value of a new run, so it is a new distinct value. Writing it to `nums[write]` appends it to the kept prefix, and advancing `write` restores the invariant.

#### Termination

When the loop finishes, every original array position has been processed.

By the invariant:

```text
nums[0:write]
```

contains exactly all distinct values from the original array, in sorted order, with no duplicates.

Therefore `write` is exactly the required length `k`, and the array prefix satisfies the problem requirement.

---

### 11. Complexity

Each element is read once.

Each unique value after the first is written once.

So:

```text
Time:  O(n)
Space: O(1)
```

The space is constant because the algorithm uses only two integer pointers and modifies the input array directly.

---

### 12. Common Pitfalls

- Returning the modified array instead of the new length `k`.
- Trying to delete elements from the list while iterating. That is slower and makes indices harder to reason about.
- Forgetting that values after `k` are irrelevant. They do not need to be cleared, replaced with underscores, or removed.
- Comparing against the wrong value after overwriting. The safest invariant-based comparison is against `nums[write - 1]`, the last kept unique value.
- Starting `write` at `0` without special handling. For a non-empty array, the first element is already the first unique value, so `write = 1` is the clean initialization.
- Missing the empty-array case in a general implementation.
- Using a set and losing the reason sorted order matters. A set can detect uniqueness but uses extra space and does not teach the in-place sorted-run structure.

---

### 13. First-Principles Summary

This problem is not about deleting values from the middle of an array. It is about constructing a valid prefix.

The sorted input gives the decisive fact:

```text
All copies of the same value appear together.
```

So a value should be kept exactly when it starts a new run.

The in-place requirement is handled by treating the beginning of `nums` as the output buffer:

```text
nums[0:write] = unique values seen so far
```

The scan pointer discovers values; the write pointer records only the first value of each run. When the scan ends, `write` is both the next free output position and the number of unique values.

## Implementation

See `solutions/array_string/p026_remove_duplicates_from_sorted_array.py`.

## Tests

See `tests/array_string/test_p026_remove_duplicates_from_sorted_array.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 1, 2]}`
- Output: `'2, nums = [1,2,_]'`

### Example 2
- Input: `{'nums': [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]}`
- Output: `'5, nums = [0,1,2,3,4,_,_,_,_,_]'`

## Follow-up Practice
- Trace the invariant after each index.
- Test empty/singleton/boundary inputs.
- Compare a brute-force version with the optimized invariant.
