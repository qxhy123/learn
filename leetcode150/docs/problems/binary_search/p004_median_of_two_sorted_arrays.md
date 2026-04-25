# 4. Median of Two Sorted Arrays

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/median-of-two-sorted-arrays/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two arrays, `nums1` and `nums2`. Each one is already sorted in nondecreasing order.

The task is to return the median of all values from both arrays together, as if the two arrays had first been merged into one sorted array.

For example:

```text
nums1 = [1, 3]
nums2 = [2]
```

The combined sorted order is:

```text
[1, 2, 3]
```

The middle value is `2`, so the answer is `2.0`.

When the total number of values is even, there is no single middle value. The median is the average of the two middle values:

```text
nums1 = [1, 2]
nums2 = [3, 4]

combined = [1, 2, 3, 4]
median = (2 + 3) / 2 = 2.5
```

The important constraint is time. A full merge is linear, but this problem asks for logarithmic time. So the real question is:

> How can we find the middle value or middle pair of the combined sorted order without building that combined array?

### 2. Start From the Baseline: Merge Then Read the Middle

The most direct solution is the merge step from merge sort:

```python
merged = []
i = 0
j = 0

while i < len(nums1) or j < len(nums2):
    if j == len(nums2) or (i < len(nums1) and nums1[i] <= nums2[j]):
        merged.append(nums1[i])
        i += 1
    else:
        merged.append(nums2[j])
        j += 1
```

Then take the middle element, or average the two middle elements.

This is correct because it explicitly constructs the same sorted order whose median we need. But it costs:

```text
Time:  O(m + n)
Space: O(m + n)
```

You can reduce the extra space by merging only until the middle, but the time is still `O(m + n)`. To reach logarithmic time, we need to avoid scanning most of the values.

### 3. The Key Observation: A Median Is a Partition Boundary

The median is determined by a split between the lower half and the upper half of the combined sorted order.

For an odd total length, the lower half can contain one extra value:

```text
combined = [1, 2, 3]
left     = [1, 2]
right    = [3]
median   = max(left) = 2
```

For an even total length, the two halves have the same size:

```text
combined = [1, 2, 3, 4]
left     = [1, 2]
right    = [3, 4]
median   = (max(left) + min(right)) / 2
```

So we do not need every position in the merged array. We only need a valid partition where:

1. the left side has the correct number of values, and
2. every left-side value is `<=` every right-side value.

Let:

```text
total = len(nums1) + len(nums2)
left_size = (total + 1) // 2
```

The `+1` is deliberate. When `total` is odd, the left side owns the extra element, so the median is simply the largest value on the left.

### 4. Turning the Split Into Two Array Cuts

Suppose we take `i` values from one array for the left side. Then the number of values we must take from the other array is forced:

```text
j = left_size - i
```

That gives two cuts:

```text
A: [ ... i values ... | ... remaining values ... ]
B: [ ... j values ... | ... remaining values ... ]
```

The size condition is now automatic:

```text
i + j = left_size
```

Only the ordering condition remains.

Because `A` and `B` are individually sorted, each array is already internally valid across its own cut:

```text
A's left side <= A's right side
B's left side <= B's right side
```

The only possible violations are cross-array violations.

Name the four boundary values:

```text
A_left  = A[i - 1]
A_right = A[i]
B_left  = B[j - 1]
B_right = B[j]
```

The partition is valid exactly when:

```text
A_left <= B_right
B_left <= A_right
```

If both are true, every value on the combined left side is `<=` every value on the combined right side.

### 5. Empty Sides and Sentinel Values

A cut is allowed at the beginning or end of an array.

If `i == 0`, then `A` contributes nothing to the left side, so `A_left` does not exist. If `i == len(A)`, then `A` contributes nothing to the right side, so `A_right` does not exist.

Use sentinel values to keep the partition logic uniform:

```text
A_left  = -infinity if i == 0
A_right = +infinity if i == len(A)
B_left  = -infinity if j == 0
B_right = +infinity if j == len(B)
```

This matches the meaning of an empty side:

- an empty left side has no maximum, so it behaves like `-infinity`;
- an empty right side has no minimum, so it behaves like `+infinity`.

With these sentinels, the same two checks handle normal cuts, edge cuts, and one-empty-array cases.

### 6. Why Binary Search Works

Search only the shorter array. Call it `A`; call the longer array `B`.

We binary search the number `i`, meaning:

```text
i = how many values A contributes to the combined left side
```

Once `i` is chosen, `j = left_size - i` is forced.

Now look at the two ways a candidate partition can fail.

#### Case 1: `A_left > B_right`

The last value taken from `A` is too large to be on the left side because it is greater than the first value on `B`'s right side.

So we took too many values from `A`.

The fix is to move the cut in `A` left:

```text
high = i - 1
```

#### Case 2: `B_left > A_right`

The last value taken from `B` is too large to be on the left side because it is greater than the first value on `A`'s right side.

Since `j = left_size - i`, taking too many from `B` means taking too few from `A`.

The fix is to move the cut in `A` right:

```text
low = i + 1
```

These directions are monotonic. A cut that is too far right in `A` cannot be fixed by moving even farther right, and a cut that is too far left in `A` cannot be fixed by moving even farther left. That monotonic repair direction is the reason binary search applies.

### 7. The Search Invariant

Maintain an inclusive search interval:

```text
low <= i <= high
```

where each `i` is a possible number of values to take from the shorter array `A`.

The invariant is:

> If the correct partition has not been found yet, some valid cut `i` still lies inside `[low, high]`.

Each iteration tries the middle cut. If `A_left > B_right`, every cut at `i` or to the right is invalid, so the valid cut must be left of `i`. If `B_left > A_right`, every cut at `i` or to the left is invalid, so the valid cut must be right of `i`.

Thus every update discards only impossible cuts and preserves the invariant.

### 8. Detailed Algorithm

1. Let `A` be the shorter input array and `B` be the longer input array.
2. Compute `total = len(A) + len(B)`.
3. Compute `left_size = (total + 1) // 2`.
4. Binary search `i` in the inclusive range `[0, len(A)]`.
5. For each `i`, compute `j = left_size - i`.
6. Read `A_left`, `A_right`, `B_left`, and `B_right`, using `-infinity` and `+infinity` for empty sides.
7. If `A_left <= B_right` and `B_left <= A_right`, the partition is valid.
8. If `total` is odd, return `max(A_left, B_left)`.
9. If `total` is even, return `(max(A_left, B_left) + min(A_right, B_right)) / 2`.
10. If `A_left > B_right`, move left with `high = i - 1`.
11. Otherwise, move right with `low = i + 1`.

### 9. Walkthrough: Odd Total Length

Use the first official example:

```text
nums1 = [1, 3]
nums2 = [2]
```

Search the shorter array:

```text
A = [2]
B = [1, 3]
total = 3
left_size = (3 + 1) // 2 = 2
```

Initial range:

```text
low = 0
high = 1
```

Try `i = 0`:

```text
j = 2

A: [ | 2]
B: [1, 3 | ]

A_left  = -inf
A_right = 2
B_left  = 3
B_right = +inf
```

The checks are:

```text
A_left <= B_right  -> -inf <= +inf  true
B_left <= A_right  -> 3 <= 2        false
```

`B_left` is too large compared with `A_right`, so we need more values from `A` on the left. Move right:

```text
low = 1
```

Try `i = 1`:

```text
j = 1

A: [2 | ]
B: [1 | 3]

A_left  = 2
A_right = +inf
B_left  = 1
B_right = 3
```

The checks are:

```text
A_left <= B_right  -> 2 <= 3     true
B_left <= A_right  -> 1 <= +inf  true
```

The partition is valid. Since `total` is odd, return the largest value on the left:

```text
max(2, 1) = 2
```

### 10. Walkthrough: Even Total Length

Use the second official example:

```text
nums1 = [1, 2]
nums2 = [3, 4]
```

Both arrays have the same length, so keep this order:

```text
A = [1, 2]
B = [3, 4]
total = 4
left_size = (4 + 1) // 2 = 2
```

Try `i = 1`:

```text
j = 1

A: [1 | 2]
B: [3 | 4]

A_left  = 1
A_right = 2
B_left  = 3
B_right = 4
```

The checks are:

```text
A_left <= B_right  -> 1 <= 4  true
B_left <= A_right  -> 3 <= 2  false
```

The cut in `A` is too far left, so move right.

Try `i = 2`:

```text
j = 0

A: [1, 2 | ]
B: [ | 3, 4]

A_left  = 2
A_right = +inf
B_left  = -inf
B_right = 3
```

The checks are both true:

```text
2 <= 3
-inf <= +inf
```

The partition is valid. Since `total` is even, average the two middle boundary values:

```text
left_middle  = max(2, -inf) = 2
right_middle = min(+inf, 3) = 3
median = (2 + 3) / 2 = 2.5
```

### 11. Pseudocode

```python
def findMedianSortedArrays(nums1, nums2):
    A = nums1
    B = nums2

    if len(A) > len(B):
        A, B = B, A

    m = len(A)
    n = len(B)
    total = m + n
    left_size = (total + 1) // 2

    low = 0
    high = m

    while low <= high:
        i = (low + high) // 2
        j = left_size - i

        A_left = float("-inf") if i == 0 else A[i - 1]
        A_right = float("inf") if i == m else A[i]
        B_left = float("-inf") if j == 0 else B[j - 1]
        B_right = float("inf") if j == n else B[j]

        if A_left <= B_right and B_left <= A_right:
            if total % 2 == 1:
                return float(max(A_left, B_left))

            return (max(A_left, B_left) + min(A_right, B_right)) / 2

        if A_left > B_right:
            high = i - 1
        else:
            low = i + 1
```

### 12. Why This Is Correct

The algorithm always chooses `i` values from `A` and `j = left_size - i` values from `B`, so the combined left side always has exactly `left_size` values.

When the algorithm accepts a partition, both cross-boundary conditions hold:

```text
A_left <= B_right
B_left <= A_right
```

Since each input array is already sorted internally, these two cross checks imply that every value in the combined left side is `<=` every value in the combined right side. Therefore the partition is exactly the lower-half/upper-half split of the fully merged sorted order.

If `total` is odd, the left side has one extra value, so the median is the largest value on the left: `max(A_left, B_left)`. If `total` is even, the median is the average of the largest left value and the smallest right value: `(max(A_left, B_left) + min(A_right, B_right)) / 2`.

It remains to justify the search updates. If `A_left > B_right`, the cut in `A` has taken too many values from `A`; no cut farther right can fix that, so the algorithm safely discards the right half. If `B_left > A_right`, the cut in `A` has taken too few values from `A`; no cut farther left can fix that, so the algorithm safely discards the left half.

Thus the search invariant is preserved, the interval strictly shrinks, and the algorithm eventually finds the valid partition and returns the correct median.

### 13. Complexity

The algorithm binary searches only the shorter array.

If `m = len(nums1)` and `n = len(nums2)`, the number of iterations is:

```text
O(log(min(m, n)))
```

Each iteration performs constant work, so:

```text
Time:  O(log(min(m, n)))
Space: O(1)
```

### 14. Common Pitfalls

- **Searching the longer array**: the forced `j` cut can become awkward or invalid. Search the shorter array.
- **Forgetting that cuts can be empty**: `i = 0`, `i = len(A)`, `j = 0`, and `j = len(B)` are valid partitions and need sentinel handling.
- **Using `<` instead of `<=`**: duplicate values are allowed, and equal boundary values form a valid partition.
- **Using `total // 2` without adjusting odd length**: `(total + 1) // 2` lets the left side own the extra value, simplifying the median formula.
- **Averaging the wrong values**: for even length, average `max(left boundaries)` and `min(right boundaries)`.
- **Trying to binary search the median value itself**: the stable search target is the partition index, not a particular number.

### 15. First-Principles Summary

The median of two sorted arrays is determined by the boundary between the lower half and the upper half of the combined sorted order.

Instead of merging the arrays, choose how many values the shorter array contributes to the lower half. The longer array's contribution is then forced. A partition is correct exactly when the two cross-boundary inequalities hold:

```text
A_left <= B_right
B_left <= A_right
```

If the first inequality fails, the cut in `A` is too far right. If the second fails, the cut in `A` is too far left. Those monotonic failure directions turn the partition search into a binary search.

## Implementation
See `solutions/binary_search/p004_median_of_two_sorted_arrays.py`.

## Tests
See `tests/binary_search/test_p004_median_of_two_sorted_arrays.py`.

## Examples

### Example 1
- Input: `{'nums1': [1, 3], 'nums2': [2]}`
- Output: `2.0`

### Example 2
- Input: `{'nums1': [1, 2], 'nums2': [3, 4]}`
- Output: `2.5`

## Follow-up Practice
- Trace the algorithm when one array is empty, such as `nums1 = []`, `nums2 = [1]`.
- Trace a case with duplicates, such as `nums1 = [1, 2, 2]`, `nums2 = [2, 3]`.
- Before coding, write down what `i`, `j`, `A_left`, `A_right`, `B_left`, and `B_right` mean.
- Explain why `A_left > B_right` means the cut in `A` must move left.
