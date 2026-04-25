# 162. Find Peak Element

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/find-peak-element/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer array `nums`, return the index of any peak element.

A peak element is an element that is strictly greater than its immediate neighbors.

For an index `i` in the middle of the array, this means:

```text
nums[i - 1] < nums[i] > nums[i + 1]
```

The problem also defines the values outside the array as negative infinity:

```text
nums[-1] = -infinity
nums[n]  = -infinity
```

That means the first element can be a peak if it is greater than the second element, and the last element can be a peak if it is greater than the second-to-last element.

For example:

```text
nums = [1, 2, 3, 1]
```

Index `2` is a peak because:

```text
nums[1] < nums[2] > nums[3]
2       < 3       > 1
```

So the answer is:

```text
2
```

For another example:

```text
nums = [1, 2, 1, 3, 5, 6, 4]
```

There are actually two peaks:

```text
index 1: value 2, because 1 < 2 > 1
index 5: value 6, because 5 < 6 > 4
```

The problem allows returning any valid peak index. The example output is `5`, but `1` would also describe a peak.

The real problem is:

> Find one index where the array rises into that element and falls away from it, treating both outside boundaries as smaller than every real element.

The follow-up expectation is to solve it in `O(log n)` time, so a linear scan is not enough for the intended solution.

---

### 2. Start From the Brute Force Idea

The direct way is to check every index.

For each index `i`, compare it with its left and right neighbors. Because the outside boundaries are negative infinity, edge indices need special handling.

Conceptually:

```python
for i in range(len(nums)):
    left_value = float("-inf") if i == 0 else nums[i - 1]
    right_value = float("-inf") if i == len(nums) - 1 else nums[i + 1]

    if left_value < nums[i] and nums[i] > right_value:
        return i
```

This is correct because it directly tests the definition of a peak.

Complexity:

```text
Time:  O(n)
Space: O(1)
```

But the problem asks for logarithmic time, so we need to avoid checking every element.

The key question becomes:

> Can one local comparison tell us which half of the array still must contain a peak?

For this problem, yes.

---

### 3. The Key Observation: Follow the Slope

Look at two adjacent elements:

```text
nums[mid] and nums[mid + 1]
```

There are two possibilities.

#### Case 1: The slope goes up

```text
nums[mid] < nums[mid + 1]
```

The array is rising from `mid` to `mid + 1`.

That means there must be at least one peak somewhere on the right side, in the range:

```text
[mid + 1, right]
```

Why?

Start at `mid + 1` and move right.

- If the values eventually go down, then the point where they first turn from rising to falling is a peak.
- If the values never go down, then the array keeps rising all the way to `right`; then `right` is a peak relative to the search interval's right boundary, and ultimately the real array boundary is `-infinity` if we reach the end.

So when the local slope goes up, we can safely discard `mid` and everything to its left for the purpose of finding at least one peak.

#### Case 2: The slope goes down

```text
nums[mid] > nums[mid + 1]
```

The array is falling from `mid` to `mid + 1`.

That means there must be at least one peak somewhere on the left side, including `mid`, in the range:

```text
[left, mid]
```

Why include `mid`?

Because `mid` is already greater than its right neighbor. If it is also greater than its left neighbor, then `mid` itself is a peak. If not, then the array rises into `mid` from the left, and following that rising direction left-to-right still guarantees a peak before or at `mid`.

So when the local slope goes down, we can safely discard everything strictly to the right of `mid`.

The important idea is not that the whole array is sorted. It is not sorted.

The important idea is this local guarantee:

```text
An upward edge points toward some peak on the right.
A downward edge points toward some peak on the left, possibly at mid.
```

This is enough structure for binary search.

---

### 4. Why a Peak Is Guaranteed to Exist

Before designing the invariant, it helps to know why the search always has an answer.

Because the boundaries outside the array are `-infinity`, a non-empty array must have at least one peak.

Think about walking from left to right:

- If `nums[0] > nums[1]`, then index `0` is a peak because the left outside value is `-infinity`.
- Otherwise, the array starts by rising.
- If the array later falls, the first element before the fall is a peak.
- If the array never falls, the last element is a peak because the right outside value is `-infinity`.

The constraint `nums[i] != nums[i + 1]` is also important. Adjacent equal values would create flat plateaus, and then the strict peak definition would require more careful handling. Here every adjacent comparison is either strictly up or strictly down.

---

### 5. The Search Invariant

Maintain an inclusive search interval:

```text
[left, right]
```

The invariant is:

```text
There is at least one peak index inside nums[left:right + 1].
```

At the beginning:

```text
left = 0
right = n - 1
```

The invariant is true because the whole array has at least one peak.

On each iteration, choose:

```text
mid = (left + right) // 2
```

The loop condition will be:

```text
while left < right
```

This detail matters. Since `left < right`, `mid` is strictly less than `right`, so `mid + 1` is always a valid index.

Then compare:

```text
nums[mid] and nums[mid + 1]
```

If:

```text
nums[mid] < nums[mid + 1]
```

then a peak exists in:

```text
[mid + 1, right]
```

So update:

```text
left = mid + 1
```

The invariant is preserved.

Otherwise:

```text
nums[mid] > nums[mid + 1]
```

a peak exists in:

```text
[left, mid]
```

So update:

```text
right = mid
```

The invariant is preserved again.

Eventually the interval shrinks to one index:

```text
left == right
```

Because the invariant says the interval contains a peak, and the interval has only one index left, that index must be a peak.

Return:

```text
left
```

---

### 6. Detailed Algorithm

1. Set `left = 0`.
2. Set `right = len(nums) - 1`.
3. While `left < right`:
   1. Compute `mid = (left + right) // 2`.
   2. Compare `nums[mid]` with `nums[mid + 1]`.
   3. If `nums[mid] < nums[mid + 1]`, move rightward:

      ```text
      left = mid + 1
      ```

   4. Otherwise, move leftward while keeping `mid`:

      ```text
      right = mid
      ```

4. Return `left`.

The algorithm never explicitly checks whether `nums[mid]` is a peak. That is intentional.

Instead of asking:

```text
Is mid the answer?
```

it asks:

```text
Which side is guaranteed to contain an answer?
```

That is the first-principles shift that makes the solution logarithmic.

---

### 7. Example Walkthrough: `nums = [1, 2, 3, 1]`

Start:

```text
nums  = [1, 2, 3, 1]
index =  0  1  2  3

left = 0
right = 3
```

The invariant says there is at least one peak in `[0, 3]`.

#### Iteration 1

```text
mid = (0 + 3) // 2 = 1
nums[mid]     = nums[1] = 2
nums[mid + 1] = nums[2] = 3
```

Compare:

```text
2 < 3
```

The slope goes up from index `1` to index `2`, so a peak must exist to the right of `mid`.

Update:

```text
left = mid + 1 = 2
right = 3
```

Now the interval is `[2, 3]`.

#### Iteration 2

```text
mid = (2 + 3) // 2 = 2
nums[mid]     = nums[2] = 3
nums[mid + 1] = nums[3] = 1
```

Compare:

```text
3 > 1
```

The slope goes down from index `2` to index `3`, so a peak must exist at `mid` or to its left within the current interval.

Update:

```text
left = 2
right = mid = 2
```

Now:

```text
left == right == 2
```

The remaining interval contains one index, and the invariant says it contains a peak. Return:

```text
2
```

Check against the definition:

```text
nums[1] < nums[2] > nums[3]
2       < 3       > 1
```

Index `2` is a peak.

---

### 8. Example Walkthrough: `nums = [1, 2, 1, 3, 5, 6, 4]`

Start:

```text
nums  = [1, 2, 1, 3, 5, 6, 4]
index =  0  1  2  3  4  5  6

left = 0
right = 6
```

#### Iteration 1

```text
mid = (0 + 6) // 2 = 3
nums[mid]     = nums[3] = 3
nums[mid + 1] = nums[4] = 5
```

Compare:

```text
3 < 5
```

The slope goes up, so search right:

```text
left = 4
right = 6
```

#### Iteration 2

```text
mid = (4 + 6) // 2 = 5
nums[mid]     = nums[5] = 6
nums[mid + 1] = nums[6] = 4
```

Compare:

```text
6 > 4
```

The slope goes down, so keep `mid` and search left within the current interval:

```text
left = 4
right = 5
```

#### Iteration 3

```text
mid = (4 + 5) // 2 = 4
nums[mid]     = nums[4] = 5
nums[mid + 1] = nums[5] = 6
```

Compare:

```text
5 < 6
```

The slope goes up, so search right:

```text
left = 5
right = 5
```

Now `left == right`, so return:

```text
5
```

Check:

```text
nums[4] < nums[5] > nums[6]
5       < 6       > 4
```

Index `5` is a peak.

Notice that index `1` is also a peak, but the binary search followed the slopes to index `5`. Since the problem allows any peak, this is correct.

---

### 9. Code

```python
class Solution:
    def findPeakElement(self, nums: list[int]) -> int:
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid

        return left
```

Equivalent pseudocode:

```text
left = 0
right = n - 1

while left < right:
    mid = floor((left + right) / 2)

    if nums[mid] < nums[mid + 1]:
        left = mid + 1
    else:
        right = mid

return left
```

---

### 10. Why This Code Is Correct

We prove correctness using the search invariant.

#### Invariant

At the start of every loop iteration, the interval:

```text
[left, right]
```

contains at least one peak index.

#### Initialization

Initially:

```text
left = 0
right = n - 1
```

The interval is the entire array. A peak must exist in the entire array because the outside boundaries are `-infinity` and adjacent elements are not equal.

So the invariant is true before the first iteration.

#### Maintenance

Assume the invariant is true at the start of an iteration.

Because the loop condition is `left < right`, the midpoint satisfies:

```text
mid < right
```

Therefore `mid + 1` is inside the current interval and can be compared safely.

If:

```text
nums[mid] < nums[mid + 1]
```

then the slope from `mid` to `mid + 1` goes upward. Following that upward direction guarantees at least one peak in `[mid + 1, right]`. The algorithm sets:

```text
left = mid + 1
```

so the new interval still contains a peak.

If instead:

```text
nums[mid] > nums[mid + 1]
```

then the slope from `mid` to `mid + 1` goes downward. Following that downward edge backward guarantees at least one peak in `[left, mid]`; `mid` itself may be the peak. The algorithm sets:

```text
right = mid
```

so the new interval still contains a peak.

In both cases, the invariant is preserved.

#### Termination

Each iteration strictly shrinks the interval.

- If `left = mid + 1`, the left boundary moves right.
- If `right = mid`, the right boundary moves left.

Since the interval length is finite, eventually:

```text
left == right
```

At that point the interval contains exactly one index.

By the invariant, that one-index interval contains a peak.

Therefore `left` is a peak index, and returning `left` is correct.

---

### 11. Why It Is `O(log n)`

Each iteration keeps only one side of the current interval.

The interval length is roughly halved each time:

```text
n, n/2, n/4, n/8, ...
```

After about `log2(n)` iterations, only one index remains.

Each iteration does constant work:

```text
compute mid
compare nums[mid] and nums[mid + 1]
move one boundary
```

Complexity:

```text
Time:  O(log n)
Space: O(1)
```

---

### 12. Common Pitfalls

#### Pitfall 1: Trying to binary search by comparing with both neighbors

It is tempting to write code that checks whether `mid` is a peak directly:

```python
if nums[mid - 1] < nums[mid] > nums[mid + 1]:
    return mid
```

This creates boundary problems at `mid = 0` and `mid = n - 1`, and it does not by itself explain which side to keep when `mid` is not a peak.

The cleaner decision is the adjacent slope comparison:

```python
nums[mid] < nums[mid + 1]
```

That one comparison always tells us a side that must contain a peak.

#### Pitfall 2: Using `while left <= right`

This solution is designed around:

```python
while left < right:
```

That guarantees `mid + 1` is valid.

If you use `while left <= right`, then eventually `mid` can equal `right`, and `nums[mid + 1]` may go out of bounds.

#### Pitfall 3: Updating `right = mid - 1`

When:

```text
nums[mid] > nums[mid + 1]
```

`mid` itself might be the peak.

So this update is wrong:

```python
right = mid - 1
```

It can discard the answer.

The correct update is:

```python
right = mid
```

#### Pitfall 4: Thinking the array must be sorted

The array is not sorted, and it does not need to be sorted.

Binary search works here because every adjacent slope points toward a side where a peak is guaranteed to exist. This is weaker than sorted order, but still strong enough to discard half the search interval.

#### Pitfall 5: Forgetting that any peak is acceptable

Some arrays have multiple peaks.

For:

```text
[1, 2, 1, 3, 5, 6, 4]
```

both indices `1` and `5` are peaks.

The algorithm may return one valid peak while a test example shows another. A correct judge should accept any valid peak index, even if the scaffolded tests in this repository currently compare against the official example output exactly.

#### Pitfall 6: Not accounting for a single-element array

If:

```text
nums = [10]
```

then index `0` is a peak because both outside neighbors are `-infinity`.

The algorithm handles this naturally:

```text
left = 0
right = 0
```

The loop never runs, and it returns `0`.

---

### 13. First-Principles Summary

This problem follows from these basic facts:

```text
1. A peak is greater than both immediate neighbors.
2. The outside boundaries count as negative infinity.
3. Therefore every non-empty input has at least one peak.
4. Adjacent elements are never equal, so every local edge is either rising or falling.
5. If nums[mid] < nums[mid + 1], a peak must exist to the right.
6. If nums[mid] > nums[mid + 1], a peak must exist at mid or to the left.
7. Keeping only a side that must contain a peak preserves the search invariant.
8. Once the interval shrinks to one index, that index must be a peak.
```

In one sentence:

> Binary search on the local slope: move toward the side that must contain a peak, preserving an interval that always contains at least one valid answer.

## Implementation

See `solutions/binary_search/p162_find_peak_element.py`.

## Tests

See `tests/binary_search/test_p162_find_peak_element.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3, 1]}`
- Output: `2`

### Example 2
- Input: `{'nums': [1, 2, 1, 3, 5, 6, 4]}`
- Output: `5`

## Follow-up Practice
- Trace the invariant on arrays where the peak is at the beginning, middle, and end.
- Explain why `right = mid` is necessary when the slope goes down.
- Compare the `O(n)` scan with the `O(log n)` slope-based binary search.
