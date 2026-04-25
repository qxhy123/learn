# 215. Kth Largest Element in an Array

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/kth-largest-element-in-an-array/
- Official Group: Heap
- Pattern Group: Heap
- Patterns: heap

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an unsorted array `nums` and an integer `k`, return the `k`th largest value in the array.

Important detail:

```text
The kth largest element is based on sorted position, not distinct value rank.
```

So duplicates count separately.

For example:

```text
nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
k = 4
```

If we sort in descending order:

```text
[6, 5, 5, 4, 3, 3, 2, 2, 1]
```

The 4th largest element is:

```text
4
```

The two `5`s occupy the 2nd and 3rd positions. We do not collapse them into one distinct value.

So the problem is:

> Find the value that would appear at index `k - 1` if the array were sorted from largest to smallest, without necessarily doing all the work of sorting the entire array.

---

### 2. Start From the Brute Force Baseline

The most direct solution is full sorting:

```python
nums.sort(reverse=True)
return nums[k - 1]
```

This is easy to reason about.

After sorting descending:

```text
index 0     -> largest
index 1     -> 2nd largest
...
index k - 1 -> kth largest
```

Correctness is immediate because sorting puts every value in the exact order we need.

But full sorting answers more questions than the problem asks.

If `nums` has one million elements and `k = 3`, sorting computes the complete order of all one million elements even though we only need the third position. The relative order of the smaller 999,997 elements does not matter.

Full sorting costs:

```text
Time:  O(n log n)
Space: depends on language/sort implementation
```

The first-principles question is:

> Can we keep only the information needed to identify the kth largest value?

Yes. Two common ways are:

1. Keep the best `k` candidates with a min-heap.
2. Partition the array so the target sorted position lands in its final place.

The heap approach is simple, stable, and works naturally in streaming settings. The partition approach can be faster on average but is more delicate.

---

### 3. Key Observation: Only the Top k Values Matter

The kth largest value is the smallest value among the largest `k` values.

Example:

```text
nums = [3, 2, 1, 5, 6, 4]
k = 2
```

The largest `2` values are:

```text
[6, 5]
```

The answer is the smallest value inside that group:

```text
5
```

This gives a useful invariant:

```text
Maintain exactly the k largest values seen so far.
The answer after scanning all numbers is the smallest value in that maintained group.
```

A min-heap is perfect for this because it can expose the smallest value among the kept candidates in `O(1)` time and remove it in `O(log k)` time.

Python's `heapq` is a min-heap, which means:

```text
heap[0] is always the smallest item currently in the heap.
```

If the heap contains the current `k` largest values seen so far, then `heap[0]` is the kth largest among the values seen so far.

---

### 4. Heap Invariant

Process the array from left to right.

Maintain a min-heap `heap` with this invariant:

```text
After processing some prefix of nums, heap contains the k largest values from that prefix,
or all values from the prefix if fewer than k values have been seen.
```

Because `heap` is a min-heap:

```text
heap[0] is the weakest candidate among the kept top-k values.
```

When a new number `x` arrives, there are two cases.

#### Case 1: Heap Has Fewer Than k Values

We have not yet collected enough candidates.

```text
push x
```

No value should be discarded yet.

#### Case 2: Heap Already Has k Values

Now `heap[0]` is the current kth largest among the processed values.

If:

```text
x <= heap[0]
```

then `x` is not bigger than the weakest kept candidate. It cannot belong to the top `k` values seen so far, so we ignore it.

If:

```text
x > heap[0]
```

then `x` deserves to enter the top `k`. To keep the heap size at exactly `k`, remove the weakest kept candidate and insert `x`.

In Python this can be done as:

```python
heapq.heapreplace(heap, x)
```

or equivalently:

```python
heapq.heappop(heap)
heapq.heappush(heap, x)
```

After this replacement, the heap again contains exactly the `k` largest values from the processed prefix.

---

### 5. Detailed Heap Algorithm

1. Create an empty min-heap.
2. For each number `x` in `nums`:
   - If the heap has fewer than `k` elements, push `x`.
   - Otherwise, compare `x` with `heap[0]`.
   - If `x` is larger than `heap[0]`, replace `heap[0]` with `x`.
   - If `x` is not larger, ignore it.
3. After all numbers are processed, return `heap[0]`.

Pseudocode:

```python
function findKthLargest(nums, k):
    heap = empty min-heap

    for x in nums:
        if size(heap) < k:
            push heap x
        else if x > heap[0]:
            replace heap[0] with x

    return heap[0]
```

Python-style code:

```python
import heapq

class Solution:
    def findKthLargest(self, nums: list[int], k: int) -> int:
        heap: list[int] = []

        for value in nums:
            if len(heap) < k:
                heapq.heappush(heap, value)
            elif value > heap[0]:
                heapq.heapreplace(heap, value)

        return heap[0]
```

A compact variation is:

```python
import heapq

class Solution:
    def findKthLargest(self, nums: list[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)

        for value in nums[k:]:
            if value > heap[0]:
                heapq.heapreplace(heap, value)

        return heap[0]
```

Both versions maintain the same invariant.

---

### 6. Walkthrough: Example 1

Input:

```text
nums = [3, 2, 1, 5, 6, 4]
k = 2
```

We maintain the largest `2` values seen so far.

Start:

```text
heap = []
```

Read `3`:

```text
heap has fewer than 2 values
push 3
heap = [3]
```

Read `2`:

```text
heap has fewer than 2 values
push 2
heap = [2, 3]
```

The heap contains `{2, 3}`. The smallest kept value is `2`.

Read `1`:

```text
heap[0] = 2
1 <= 2
ignore 1
heap = [2, 3]
```

`1` cannot be in the top two values seen so far.

Read `5`:

```text
heap[0] = 2
5 > 2
replace 2 with 5
heap = [3, 5]
```

The top two values seen so far are now `{3, 5}`.

Read `6`:

```text
heap[0] = 3
6 > 3
replace 3 with 6
heap = [5, 6]
```

The top two values seen so far are now `{5, 6}`.

Read `4`:

```text
heap[0] = 5
4 <= 5
ignore 4
heap = [5, 6]
```

After scanning every value, the heap contains the largest two values from the whole array:

```text
[5, 6]
```

The smallest among them is:

```text
5
```

So the answer is `5`.

---

### 7. Walkthrough: Example 2 With Duplicates

Input:

```text
nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
k = 4
```

The sorted descending order would be:

```text
[6, 5, 5, 4, 3, 3, 2, 2, 1]
```

So the answer is `4`.

Heap trace:

```text
read 3 -> heap has room -> [3]
read 2 -> heap has room -> [2, 3]
read 3 -> heap has room -> [2, 3, 3]
read 1 -> heap has room -> [1, 2, 3, 3]
```

Now the heap has four values. Its root is the weakest candidate:

```text
heap[0] = 1
```

Continue:

```text
read 2 -> 2 > 1 -> replace 1 -> heap contains {2, 2, 3, 3}
read 4 -> 4 > 2 -> replace 2 -> heap contains {2, 3, 3, 4}
read 5 -> 5 > 2 -> replace 2 -> heap contains {3, 3, 4, 5}
read 5 -> 5 > 3 -> replace 3 -> heap contains {3, 4, 5, 5}
read 6 -> 6 > 3 -> replace 3 -> heap contains {4, 5, 5, 6}
```

At the end, the heap contains the largest four values, counting duplicates:

```text
{4, 5, 5, 6}
```

The smallest value in this group is:

```text
4
```

That is the 4th largest element.

---

### 8. Why the Heap Algorithm Is Correct

We prove the invariant by induction over the scanned prefix of the array.

Invariant:

```text
After processing the first i numbers, heap contains the k largest values among those i numbers,
or all i values if i < k.
```

#### Base Case

Before processing any numbers:

```text
heap = []
```

The heap contains all values from the empty prefix, so the invariant holds.

#### Inductive Step

Assume the invariant holds before reading a new value `x`.

If the heap has fewer than `k` values, then fewer than `k` total values have been processed. Adding `x` to the heap means the heap still contains all processed values. The invariant holds.

If the heap already has `k` values, then by the induction hypothesis it contains the current top `k` values from the previous prefix. Since `heap[0]` is the smallest of those top `k` values, it is the current kth largest value.

There are two possibilities:

1. `x <= heap[0]`

   Then at least `k` previous values are greater than or equal to `x`: the values already in the heap. Therefore `x` cannot be one of the top `k` values after adding it. Ignoring `x` preserves the invariant.

2. `x > heap[0]`

   Then `x` is larger than the weakest current top-k value. The old `heap[0]` can no longer remain in the top `k`, because `x` replaces it. Removing `heap[0]` and inserting `x` produces exactly the new top `k` values. The invariant holds.

By induction, after every number has been processed, the heap contains the `k` largest values in the entire array.

The kth largest value is the smallest value among those `k` values, and a min-heap stores that value at `heap[0]`.

Therefore the algorithm returns the correct answer.

---

### 9. Complexity of the Heap Approach

Let `n = len(nums)`.

The heap never contains more than `k` elements.

Each push or replacement costs:

```text
O(log k)
```

We scan `n` values, so the total time is:

```text
O(n log k)
```

The heap stores at most `k` values, so the extra space is:

```text
O(k)
```

This is especially good when `k` is small compared with `n`.

For example, finding the 10th largest value in a huge array does not require sorting the entire array.

---

### 10. Partition / Quickselect View

There is another first-principles route: do not maintain a heap; instead, partially arrange the array until the target position is fixed.

If the array were sorted in ascending order, the kth largest element would be at index:

```text
target = len(nums) - k
```

Example:

```text
nums sorted ascending = [1, 2, 3, 4, 5, 6]
k = 2
target = 6 - 2 = 4
nums[4] = 5
```

Quickselect uses partitioning:

```text
Choose a pivot.
Move values smaller than the pivot to its left.
Move values larger than the pivot to its right.
Place the pivot in its final sorted index.
```

Partition invariant for ascending partition:

```text
all values left of pivot_index  <= pivot
nums[pivot_index]              == pivot
all values right of pivot_index >= pivot
```

After partitioning, if:

```text
pivot_index == target
```

then the pivot is exactly the kth largest value.

If:

```text
pivot_index < target
```

then the target lies to the right, so search only the right side.

If:

```text
pivot_index > target
```

then the target lies to the left, so search only the left side.

Quickselect average complexity is:

```text
Time:  O(n) average, O(n^2) worst case
Space: O(1) extra if done in-place
```

This is often the fastest practical approach, especially with randomized pivots. However, it mutates the array and has more implementation edge cases than the heap solution.

For an interview, the heap solution is often easier to present correctly; Quickselect is a strong follow-up when average linear time is requested.

---

### 11. Common Pitfalls

#### Treating kth largest as kth distinct largest

Wrong interpretation:

```text
[5, 5, 4], k = 2 -> answer 4
```

Correct interpretation:

```text
[5, 5, 4], k = 2 -> answer 5
```

Duplicates count as separate array positions.

#### Keeping a max-heap of size k

A size-`k` heap for this approach should be a min-heap, not a max-heap.

The root must be the weakest kept candidate so it can be replaced when a better value appears.

If using Python's `heapq`, no negation is needed for the min-heap top-k approach.

#### Letting the heap grow to n

If all values are pushed and then popped `k - 1` times, the solution can still work, but it uses more memory and usually more time:

```text
O(n) space instead of O(k)
```

The key optimization is to discard values that cannot belong to the top `k`.

#### Replacing on `>=` instead of `>`

Using `>=` is usually still correct for the final answer, but it performs unnecessary replacements for equal values.

The cleaner rule is:

```text
replace only when value > heap[0]
```

Equal values do not improve the kept group.

#### Off-by-one error in Quickselect

For ascending partition, the target index for kth largest is:

```text
len(nums) - k
```

not `k`, and not `k - 1`.

`k - 1` is the index only in descending sorted order.

#### Forgetting that Quickselect mutates the array

Partition-based selection rearranges `nums`. If callers expect the input order to remain unchanged, copy the array first or use the heap approach.

---

### 12. First-Principles Summary

The problem asks for one sorted position, not a fully sorted array.

Full sorting works because it creates every sorted position, but it does unnecessary work.

The heap solution focuses only on the information that can affect the answer:

```text
the largest k values seen so far
```

A min-heap of size `k` gives direct access to the weakest value in that group. Every new number is judged against that weakest kept candidate:

```text
if it is not bigger, it cannot matter
if it is bigger, it replaces the weakest candidate
```

After the scan, the heap contains exactly the largest `k` values in the array, and the smallest of them is the kth largest element.

The partition solution uses a different invariant: after partitioning around a pivot, the pivot is in its final sorted position. Repeating partition only on the side containing the target index finds the kth largest value without fully sorting.

Both approaches are built from the same first-principles idea:

> Do only enough ordering work to identify the one position the problem asks for.

## Implementation
See `solutions/heap/p215_kth_largest_element_in_an_array.py`.

## Tests
See `tests/heap/test_p215_kth_largest_element_in_an_array.py`.

## Examples

### Example 1
- Input: `{'nums': [3, 2, 1, 5, 6, 4], 'k': 2}`
- Output: `5`

### Example 2
- Input: `{'nums': [3, 2, 3, 1, 2, 4, 5, 5, 6], 'k': 4}`
- Output: `4`

## Follow-up Practice

- Trace the size-`k` min-heap by hand for `nums = [7, 10, 4, 3, 20, 15]`, `k = 3`.
- Explain why the heap root is the answer only after the heap contains exactly `k` values.
- Rewrite the solution with Quickselect and identify the ascending target index `len(nums) - k`.
- Test duplicate-heavy inputs such as `[2, 1, 2, 2, 3]` with several values of `k`.
