# 295. Find Median from Data Stream

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/find-median-from-data-stream/
- Official Group: Heap
- Pattern Group: Heap
- Patterns: heap

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

We need to design a data structure with two operations:

```text
addNum(num)      insert one new integer from the stream
findMedian()    return the median of all numbers inserted so far
```

The input is a stream, which means the numbers arrive one at a time. We are not given the whole array once and asked for one final median. Instead, after any insertion, the caller may ask for the current median immediately.

The median is the middle value after sorting the current numbers:

- If the count is odd, the median is the single middle value.
- If the count is even, the median is the average of the two middle values.

For example, after inserting:

```text
1, 2, 3
```

the sorted order is:

```text
[1, 2, 3]
```

The median is `2`.

After inserting only:

```text
1, 2
```

the sorted order is:

```text
[1, 2]
```

The median is:

```text
(1 + 2) / 2 = 1.5
```

So the real problem is:

> Maintain enough order information after every insertion to answer "what is the middle?" quickly.

The challenge is not computing a median once. The challenge is preserving the middle as the data changes.

---

### 2. The Brute Force Baseline

The most direct solution is to store every number in a list.

For each `addNum(num)`:

```text
append num to the list
```

For each `findMedian()`:

```text
sort the list
return the middle value or average of the two middle values
```

This is simple and correct because sorting gives the exact order needed to identify the median.

But it is too expensive if `findMedian()` is called many times.

If there are `n` inserted numbers:

- `addNum` costs `O(1)` to append.
- `findMedian` costs `O(n log n)` to sort.

If the stream has many alternating insertions and median queries, repeatedly sorting almost the same data wastes work.

We can improve the baseline by keeping the list sorted after every insertion. Then:

- `findMedian` is `O(1)` because the middle index is known.
- `addNum` still costs `O(n)` because inserting into the middle of an array requires shifting elements.

That is better for many queries, but still not ideal for a long stream.

The first-principles question is:

> Do we really need the entire sorted order, or only the part of the order around the middle?

---

### 3. The Key Observation

The median only depends on the boundary between the lower half and the upper half.

Suppose the current sorted numbers are:

```text
[1, 2, 3, 4, 5, 6, 7]
```

The median is `4`.

Everything below the median is:

```text
[1, 2, 3]
```

Everything above the median is:

```text
[5, 6, 7]
```

To know the median, we do not need to know that `1` comes before `2`, or that `6` comes before `7`. We only need to know:

- the largest value in the lower half, and
- the smallest value in the upper half.

For an odd number of values, one half can hold one extra element. The top of that larger half is the median.

For an even number of values, the median is the average of:

- the largest value in the lower half, and
- the smallest value in the upper half.

That suggests splitting the stream into two groups:

```text
lower half | upper half
```

with this order property:

```text
every value in lower half <= every value in upper half
```

If we can maintain this split while numbers arrive, median lookup becomes cheap.

---

### 4. Why Two Heaps Fit the Problem

We need quick access to two boundary values:

```text
max(lower half)
min(upper half)
```

A heap gives quick access to one extreme:

- a min-heap gives the smallest value,
- a max-heap gives the largest value.

Python's `heapq` is a min-heap only, so a max-heap is usually simulated by storing negative numbers.

Use two heaps:

```text
small = max-heap for the lower half
large = min-heap for the upper half
```

In Python-style notation:

```text
small stores negatives, so -small[0] is the largest lower-half value
large stores positives, so large[0] is the smallest upper-half value
```

The two heaps do not fully sort their halves. They only guarantee that each half's boundary value is available at the top.

That is exactly enough information for the median.

---

### 5. The Two-Heap Invariant

The entire solution rests on maintaining two conditions after every insertion.

#### Invariant A: Size Balance

The heaps must have almost the same size:

```text
len(small) == len(large)
```

or:

```text
len(small) == len(large) + 1
```

This version lets `small` hold the extra element when the total count is odd.

Other implementations allow either heap to be larger by one. That is also valid, but the median logic must match the chosen convention. This tutorial uses the simpler convention:

> `small` is either the same size as `large`, or one element larger.

#### Invariant B: Order Separation

Every value in `small` must be less than or equal to every value in `large`:

```text
max(small) <= min(large)
```

Using Python's negative max-heap representation:

```text
-small[0] <= large[0]
```

when both heaps are non-empty.

Together, these invariants mean the middle is always at the heap tops:

- If `small` has one extra element, `-small[0]` is the median.
- If both heaps have the same size, `(-small[0] + large[0]) / 2` is the median.

---

### 6. Detailed Algorithm

For each incoming number `num`, we need to insert it into one of the two halves and then restore the invariants.

One clean approach is:

1. Push the new number into `small`.
2. Move the largest value from `small` into `large`.
3. If `large` became bigger than `small`, move the smallest value from `large` back into `small`.

In more detail:

```text
addNum(num):
    push -num into small

    biggest_lower = pop from small
    push -biggest_lower into large

    if large has more elements than small:
        smallest_upper = pop from large
        push -smallest_upper into small
```

Why this works:

- Pushing into `small` first temporarily treats the new value as part of the lower half.
- Moving `small`'s largest value to `large` repairs the order boundary. If the new value was too large for the lower half, it gets moved upward. If some previous lower-half value is now the largest boundary value, that one moves upward instead.
- Moving one value back from `large` to `small` repairs size balance when needed.

This avoids many branching cases. We do not need to separately decide whether `num` belongs in the lower or upper half. The heap operations decide by boundary values.

Then median lookup is direct:

```text
findMedian():
    if small has more elements than large:
        return -small[0]
    else:
        return (-small[0] + large[0]) / 2
```

---

### 7. Example Walkthrough

Use the official-style operation sequence:

```text
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[],             [1],      [2],      [],           [3],      []]
```

Start with two empty heaps:

```text
small = []
large = []
```

Remember:

```text
small stores negatives
large stores normal values
```

#### Add `1`

Push into `small`:

```text
small = [-1]
large = []
```

Move largest lower-half value to `large`:

```text
small = []
large = [1]
```

Now `large` is bigger, so move its smallest value back to `small`:

```text
small = [-1]
large = []
```

Logical halves:

```text
lower = [1]
upper = []
```

Median would be `1`.

#### Add `2`

Push into `small`:

```text
small = [-2, -1]
large = []
```

Because `small` is a simulated max-heap, the largest logical value is `2`.

Move largest lower-half value to `large`:

```text
small = [-1]
large = [2]
```

Sizes are equal, so no move back is needed.

Logical halves:

```text
lower = [1]
upper = [2]
```

Now `findMedian()` returns:

```text
(1 + 2) / 2 = 1.5
```

#### Add `3`

Push into `small`:

```text
small = [-3, -1]
large = [2]
```

Move largest lower-half value to `large`:

```text
small = [-1]
large = [2, 3]
```

Now `large` is bigger, so move its smallest value back to `small`:

```text
small = [-2, -1]
large = [3]
```

Logical halves:

```text
lower = [1, 2]
upper = [3]
```

Now `findMedian()` returns:

```text
2
```

The full sorted stream is `[1, 2, 3]`, and the heap boundary gives the same middle without storing the stream in sorted order.

---

### 8. Pseudocode / Reference Code

The LeetCode interface is usually a `MedianFinder` class. In Python, the core implementation looks like this:

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap simulated with negative values
        self.large = []  # min-heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)

        largest_lower = -heapq.heappop(self.small)
        heapq.heappush(self.large, largest_lower)

        if len(self.large) > len(self.small):
            smallest_upper = heapq.heappop(self.large)
            heapq.heappush(self.small, -smallest_upper)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])

        return (-self.small[0] + self.large[0]) / 2
```

The important part is not the exact variable names. The important part is that every insertion ends with:

```text
size balance + order separation
```

Once those are true, `findMedian` is just reading the boundary values.

---

### 9. Correctness

We prove that the algorithm returns the median after every sequence of insertions.

#### Lemma 1: The size invariant holds after every insertion.

The algorithm first pushes one value into `small`, then moves one value from `small` to `large`. After those two operations, the total number of elements increased by one, but `large` may have more elements than `small`.

If `large` is larger, the algorithm moves one value from `large` back to `small`. Therefore, after rebalancing, `small` is either the same size as `large` or exactly one element larger.

So the size invariant holds.

#### Lemma 2: The order invariant holds after every insertion.

Before insertion, assume every value in `small` is less than or equal to every value in `large`.

The algorithm inserts the new value into `small`, then removes the largest logical value from `small` and inserts it into `large`.

After that move, every remaining value in `small` is less than or equal to the moved value, and the moved value is now in `large`. Since all previous `large` values were already greater than or equal to all previous `small` values, the boundary remains correctly separated.

If `large` is too large, the algorithm moves the smallest value from `large` back into `small`. Because this is the smallest upper-half value, every value left in `large` is greater than or equal to it. Adding it to `small` therefore preserves the fact that every lower-half value is less than or equal to every upper-half value.

So the order invariant holds.

#### Lemma 3: The heap tops identify the median.

By the order invariant, all values in `small` come before all values in `large` in sorted order.

By the size invariant, the split is exactly at the middle:

- If the total count is odd, `small` has one extra value, so the last value of the lower half is the middle value. That value is `max(small)`, available as `-small[0]`.
- If the total count is even, both heaps have the same size, so the two middle values are `max(small)` and `min(large)`, available as `-small[0]` and `large[0]`.

Therefore `findMedian()` returns the correct median.

#### Theorem: The data structure is correct.

After initialization, both heaps are empty, so the invariants hold trivially. Lemmas 1 and 2 show that every call to `addNum` preserves the invariants. Lemma 3 shows that whenever the invariants hold, `findMedian` returns the correct median.

Therefore, after any sequence of valid insertions, `findMedian` returns the median of all inserted numbers.

---

### 10. Complexity

Let `n` be the number of values inserted so far.

For `addNum(num)`:

- Each heap push or pop costs `O(log n)`.
- The algorithm performs a constant number of heap operations.
- Total time is `O(log n)`.

For `findMedian()`:

- The median is read from one or two heap tops.
- Total time is `O(1)`.

Space:

- Every inserted number is stored in exactly one heap.
- Total space is `O(n)`.

This is the key improvement over repeated sorting: insertions cost logarithmic time, and median queries are constant time.

---

### 11. Common Pitfalls

- Forgetting that Python's `heapq` is a min-heap. To simulate a max-heap, store `-num` and negate again when reading or moving values.
- Returning `small[0]` directly. In the negative-value max-heap representation, the logical value is `-small[0]`.
- Letting one heap become larger by more than one element. Then the split no longer identifies the middle.
- Preserving size balance but not order separation. Equal heap sizes are not enough; every lower-half value must still be less than or equal to every upper-half value.
- Averaging the wrong two values for an even count. The two middle values are `max(lower)` and `min(upper)`, not arbitrary heap elements.
- Using integer division. The median may be fractional, so the even-count case must produce a float-like result.
- Mixing conventions. If `large` is allowed to hold the extra element instead of `small`, then `findMedian` must be adjusted accordingly.

---

### 12. First-Principles Summary

The median is not a property that requires full sorted order. It only requires knowing the boundary between the lower half and the upper half.

That boundary has two useful values:

```text
largest value in lower half
smallest value in upper half
```

A max-heap is the right structure for the first value, and a min-heap is the right structure for the second value.

So the problem becomes maintaining two simple invariants:

```text
1. The heaps are balanced in size.
2. Every lower-half value is <= every upper-half value.
```

Once those invariants are true, the median is forced:

- odd count: top of the larger lower-half heap,
- even count: average of the two heap tops.

This is why the two-heap solution is not a trick. It is a direct translation of what the median fundamentally means.

## Implementation
See `solutions/heap/p295_find_median_from_data_stream.py`.

## Tests
See `tests/heap/test_p295_find_median_from_data_stream.py`.

## Examples

### Example 1
- Input: `{'raw': '["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]\n[[],[1],[2],[],[3],[]]'}`
- Output: `'See official examples'`

Expanded result sequence:

```text
MedianFinder() -> null
addNum(1)      -> null
addNum(2)      -> null
findMedian()   -> 1.5
addNum(3)      -> null
findMedian()   -> 2.0
```

### Example 2

Stream:

```text
5, 15, 1, 3
```

After each insertion:

```text
[5]             median = 5
[5, 15]         median = 10
[1, 5, 15]      median = 5
[1, 3, 5, 15]   median = 4
```

The heaps do not need to store those arrays in sorted order. They only need to preserve the split around the middle.

## Follow-up Practice

- Trace the heap contents after inserting a descending stream such as `5, 4, 3, 2, 1`.
- Reimplement the algorithm with the convention that the upper heap may hold the extra element, then adjust `findMedian` accordingly.
- Compare against the brute-force sorted-list baseline after every insertion to confirm the same median sequence.
