# 373. Find K Pairs with Smallest Sums

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
- Official Group: Heap
- Pattern Group: Heap
- Patterns: heap, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two integer arrays sorted in ascending order:

```text
nums1 = [a0, a1, a2, ...]
nums2 = [b0, b1, b2, ...]
```

A pair is formed by choosing one number from `nums1` and one number from `nums2`:

```text
[nums1[i], nums2[j]]
```

The pair's priority is its sum:

```text
nums1[i] + nums2[j]
```

The task is to return the `k` pairs with the smallest sums.

For example:

```text
nums1 = [1, 7, 11]
nums2 = [2, 4, 6]
k = 3
```

All possible pairs are:

```text
[1, 2]   sum = 3
[1, 4]   sum = 5
[1, 6]   sum = 7
[7, 2]   sum = 9
[7, 4]   sum = 11
[7, 6]   sum = 13
[11, 2]  sum = 13
[11, 4]  sum = 15
[11, 6]  sum = 17
```

The three smallest are:

```text
[[1, 2], [1, 4], [1, 6]]
```

So the real problem is:

> Produce the first `k` items from the sorted order of every cross-array pair, without explicitly building and sorting every pair when the arrays are large.

---

### 2. Start From the Brute Force Idea

The most direct solution is:

1. Generate every pair `[nums1[i], nums2[j]]`.
2. Compute its sum.
3. Sort all pairs by sum.
4. Return the first `k` pairs.

Conceptually:

```python
pairs = []

for i in range(len(nums1)):
    for j in range(len(nums2)):
        pairs.append((nums1[i] + nums2[j], nums1[i], nums2[j]))

pairs.sort()
return [[a, b] for _, a, b in pairs[:k]]
```

This is correct because it considers every possible pair.

But if:

```text
len(nums1) = m
len(nums2) = n
```

then there are:

```text
m * n
```

pairs.

The brute-force complexity is:

```text
Time:  O(mn log(mn))
Space: O(mn)
```

That is much more work than necessary when `k` is small. If we only need 3 pairs, building one million pairs is wasteful.

The better question is:

> Can we discover pairs in increasing sum order, stopping after `k` outputs?

---

### 3. The Key Observation: Sorted Arrays Create Sorted Rows

Think of all pair sums as a matrix.

Rows come from `nums1`.
Columns come from `nums2`.

For:

```text
nums1 = [1, 7, 11]
nums2 = [2, 4, 6]
```

the sum matrix is:

```text
             nums2
           2   4   6
nums1 1    3   5   7
      7    9  11  13
     11   13  15  17
```

Because `nums2` is sorted, each row is sorted left to right:

```text
nums1[i] + nums2[0] <= nums1[i] + nums2[1] <= nums1[i] + nums2[2] <= ...
```

Because `nums1` is sorted, each column is sorted top to bottom:

```text
nums1[0] + nums2[j] <= nums1[1] + nums2[j] <= nums1[2] + nums2[j] <= ...
```

This sorted grid structure is the whole reason a heap works.

The smallest pair in a row is always the first column:

```text
(nums1[i], nums2[0])
```

After we take that pair from row `i`, the next possible pair from the same row is:

```text
(nums1[i], nums2[1])
```

Then:

```text
(nums1[i], nums2[2])
```

and so on.

So each row behaves like a sorted list of pairs:

```text
row i:
(nums1[i], nums2[0]), (nums1[i], nums2[1]), (nums1[i], nums2[2]), ...
```

The problem becomes:

> Merge many sorted rows and output the first `k` elements of the merged order.

That is exactly what a min-heap is good at.

---

### 4. Why Not Push Every Cell?

A tempting heap solution is:

1. Push every pair into a min-heap.
2. Pop `k` times.

This avoids sorting explicitly, but it still stores all `m * n` pairs.

That is not the main improvement.

The important improvement is to store only the current frontier.

A frontier is the set of candidates that could be the next smallest pair.

At the start, for each useful row `i`, the smallest unreturned pair in that row is:

```text
(i, 0)
```

There is no reason to push `(i, 1)` yet, because `(i, 0)` is smaller or equal and must come before it from the same row.

Once `(i, 0)` is popped, then `(i, 1)` becomes the smallest remaining pair from row `i`, so it enters the frontier.

This is the same idea as merging sorted linked lists: keep only one current node from each list.

---

### 5. The Heap Frontier Invariant

Maintain a min-heap of triples:

```text
(sum, i, j)
```

where:

```text
sum = nums1[i] + nums2[j]
```

The invariant is:

```text
For every active row i, the heap contains exactly the smallest not-yet-output pair from that row.
```

At initialization, the smallest pair from row `i` is `(i, 0)`, so we push:

```text
(nums1[i] + nums2[0], i, 0)
```

After popping `(i, j)`, that pair is no longer available. The next smallest not-yet-output pair from the same row is `(i, j + 1)`, if `j + 1` exists.

So we push:

```text
(nums1[i] + nums2[j + 1], i, j + 1)
```

This preserves the invariant.

The heap's minimum is therefore the smallest among all rows' smallest remaining pairs. Since every other unpushed pair in a row is at least as large as that row's frontier pair, no hidden pair can be smaller than the heap minimum.

That is the crucial proof idea.

---

### 6. Why Only the First `k` Rows Are Needed

At first glance, we might push `(i, 0)` for every row `i`.

That works.

But we can push fewer rows:

```text
i = 0, 1, 2, ..., min(k, len(nums1)) - 1
```

Why is this safe?

Any row beyond index `k - 1` starts with:

```text
nums1[i] + nums2[0]
```

Because `nums1` is sorted, for `i >= k`:

```text
nums1[i] + nums2[0] >= nums1[k - 1] + nums2[0]
```

Before a row beyond `k - 1` could contribute one of the first `k` pairs, there are already at least `k` candidate pairs no larger than its first pair:

```text
(0, 0), (1, 0), (2, 0), ..., (k - 1, 0)
```

So rows after the first `k` cannot be needed for the first `k` outputs.

This limits the heap size to:

```text
min(k, len(nums1))
```

instead of `len(nums1)`.

---

### 7. Algorithm

Handle empty input first:

```text
if nums1 is empty or nums2 is empty or k == 0:
    return []
```

Then:

1. Create an empty min-heap.
2. For each row `i` from `0` to `min(k, len(nums1)) - 1`, push the first pair in that row:

```text
(nums1[i] + nums2[0], i, 0)
```

3. Create an empty result list.
4. While the heap is not empty and fewer than `k` pairs have been returned:
   - Pop the smallest heap entry `(sum, i, j)`.
   - Append `[nums1[i], nums2[j]]` to the result.
   - If `j + 1 < len(nums2)`, push the next pair from the same row:

```text
(nums1[i] + nums2[j + 1], i, j + 1)
```

5. Return the result.

The algorithm never needs to mark cells visited because it only moves right within a row. Each cell `(i, j)` can be reached from exactly one previous cell `(i, j - 1)`.

---

### 8. Example Walkthrough

Use:

```text
nums1 = [1, 7, 11]
nums2 = [2, 4, 6]
k = 3
```

The rows are:

```text
row 0: [1, 2] sum 3,  [1, 4] sum 5,  [1, 6] sum 7
row 1: [7, 2] sum 9,  [7, 4] sum 11, [7, 6] sum 13
row 2: [11, 2] sum 13, [11, 4] sum 15, [11, 6] sum 17
```

Initialize the heap with the first pair from each of the first `min(k, len(nums1)) = 3` rows:

```text
heap = [
  (3, 0, 0),   # [1, 2]
  (9, 1, 0),   # [7, 2]
  (13, 2, 0),  # [11, 2]
]
result = []
```

#### Pop 1

The smallest heap entry is:

```text
(3, 0, 0) -> [1, 2]
```

Append it:

```text
result = [[1, 2]]
```

Now row `0` has exposed its next pair `(0, 1)`:

```text
[1, 4] sum = 5
```

Push it:

```text
heap = [
  (5, 0, 1),   # [1, 4]
  (9, 1, 0),   # [7, 2]
  (13, 2, 0),  # [11, 2]
]
```

#### Pop 2

The smallest heap entry is:

```text
(5, 0, 1) -> [1, 4]
```

Append it:

```text
result = [[1, 2], [1, 4]]
```

Now row `0` exposes `(0, 2)`:

```text
[1, 6] sum = 7
```

Push it:

```text
heap = [
  (7, 0, 2),   # [1, 6]
  (9, 1, 0),   # [7, 2]
  (13, 2, 0),  # [11, 2]
]
```

#### Pop 3

The smallest heap entry is:

```text
(7, 0, 2) -> [1, 6]
```

Append it:

```text
result = [[1, 2], [1, 4], [1, 6]]
```

Now the result has `k = 3` pairs, so stop.

Final answer:

```text
[[1, 2], [1, 4], [1, 6]]
```

Notice that pairs like `[7, 4]` and `[11, 6]` were never pushed. They could not matter before earlier pairs in their rows were removed from the frontier.

---

### 9. Code

```python
from heapq import heappop, heappush
from typing import List


class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2 or k == 0:
            return []

        heap = []

        for i in range(min(k, len(nums1))):
            heappush(heap, (nums1[i] + nums2[0], i, 0))

        result = []

        while heap and len(result) < k:
            _, i, j = heappop(heap)
            result.append([nums1[i], nums2[j]])

            next_j = j + 1
            if next_j < len(nums2):
                heappush(heap, (nums1[i] + nums2[next_j], i, next_j))

        return result
```

The heap stores indices instead of values so that, after returning `[nums1[i], nums2[j]]`, the code can find the next pair in the same row: `[nums1[i], nums2[j + 1]]`.

---

### 10. Why This Code Is Correct

The proof comes from the frontier invariant.

For each active row `i`, the heap contains the leftmost pair in that row that has not already been output.

This is true at the beginning because the algorithm pushes `(i, 0)` for each initialized row, and column `0` is the smallest pair in row `i`.

Suppose the invariant is true before a pop.

The heap minimum is the smallest frontier pair across all active rows. In any particular row, every unpushed pair to the right is greater than or equal to that row's frontier pair, because `nums2` is sorted. Therefore, no unpushed pair can be smaller than the minimum heap entry.

So when the algorithm pops `(i, j)`, it really is the smallest remaining pair among all pairs that can still appear in the first `k` results.

After popping `(i, j)`, the smallest not-yet-output pair from row `i` becomes `(i, j + 1)`, if that column exists. The algorithm pushes exactly that next pair. All other active rows are unchanged. Therefore the frontier invariant is restored.

By induction, every pop returns the next pair in global nondecreasing sum order.

The algorithm stops after appending `k` pairs or after the heap becomes empty. If fewer than `k` total pairs exist, the heap eventually empties and all possible pairs have been output. Otherwise, the first `k` pops are exactly the `k` pairs with smallest sums.

Thus the returned list is correct.

---

### 11. Complexity

Let:

```text
m = len(nums1)
n = len(nums2)
h = min(k, m)
```

The heap initially contains at most `h` entries.

Each output pair requires:

```text
one heappop
possibly one heappush
```

Each heap operation costs:

```text
O(log h)
```

The algorithm outputs at most:

```text
min(k, m * n)
```

pairs.

So the complexity is:

```text
Time:  O(min(k, m * n) * log(min(k, m)))
Space: O(min(k, m))
```

In the common case where `k` is much smaller than `m * n`, this is far better than generating all pairs.

---

### 12. Common Pitfalls

#### Pitfall 1: Generating every pair

This is simple but loses the main advantage of the sorted input.

The point is not merely to use a heap. The point is to keep only the frontier of each sorted row.

#### Pitfall 2: Pushing both right and down neighbors without a visited set

Another valid mental model is a grid search from `(0, 0)` where each cell can push `(i + 1, j)` and `(i, j + 1)`.

But then the same cell can be reached in multiple ways. For example, `(1, 1)` can be reached from `(0, 1)` and from `(1, 0)`.

That version needs a `visited` set.

The row-frontier version in this tutorial only pushes rightward within the same row, so duplicates from multiple paths do not happen.

#### Pitfall 3: Forgetting to store indices

If the heap stores only:

```text
(sum, nums1[i], nums2[j])
```

then after popping, the algorithm does not know which `j` produced the pair, so it cannot push the next pair from the same row.

Store:

```text
(sum, i, j)
```

#### Pitfall 4: Initializing too many rows

Pushing all `len(nums1)` first-column pairs is correct, but unnecessary when `k < len(nums1)`.

Only the first `min(k, len(nums1))` rows can contribute to the first `k` results.

#### Pitfall 5: Assuming sums are unique

Different pairs can have the same sum, and duplicate values in the input can produce duplicate output pairs.

For example:

```text
nums1 = [1, 1, 2]
nums2 = [1, 2, 3]
k = 2
```

The answer is:

```text
[[1, 1], [1, 1]]
```

Those are two different index pairs even though the values are the same.

#### Pitfall 6: Missing empty-array and oversized-`k` cases

If either input array is empty, there are no pairs.

If `k` is larger than the total number of pairs, the loop should naturally stop when the heap becomes empty.

---

### 13. First-Principles Summary

This problem follows from these basic facts:

```text
1. Every answer pair is a cell in an m-by-n sum matrix.
2. Because both arrays are sorted, every row of that matrix is sorted left to right.
3. The first unreturned cell in a row is the only cell from that row that can currently be the global minimum.
4. A min-heap can choose the smallest among those row frontiers.
5. After a row frontier is popped, the next cell to the right becomes that row's new frontier.
6. Repeating this process emits pairs in nondecreasing sum order.
7. Stopping after k pops gives exactly the k smallest pairs.
```

In one sentence:

> Treat the pair sums as sorted rows, keep the smallest unreturned pair from each relevant row in a min-heap, and after each pop advance only that row's frontier one step to the right.

## Implementation

See `solutions/heap/p373_find_k_pairs_with_smallest_sums.py`.

## Tests

See `tests/heap/test_p373_find_k_pairs_with_smallest_sums.py`.

## Examples

### Example 1
- Input: `{'nums1': [1, 7, 11], 'nums2': [2, 4, 6], 'k': 3}`
- Output: `[[1, 2], [1, 4], [1, 6]]`

### Example 2
- Input: `{'nums1': [1, 1, 2], 'nums2': [1, 2, 3], 'k': 2}`
- Output: `[[1, 1], [1, 1]]`

## Follow-up Practice
- Trace the heap contents after every pop for `nums1 = [1, 2]`, `nums2 = [3, 4, 5]`, and `k = 4`.
- Compare the row-frontier heap approach with the grid-neighbor approach that needs a `visited` set.
- Explain why row `k` cannot be needed when the first `k` rows have already supplied `k` first-column candidates.
