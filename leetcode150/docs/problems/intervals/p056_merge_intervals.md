# 56. Merge Intervals

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/merge-intervals/
- Official Group: Intervals
- Pattern Group: Intervals
- Patterns: intervals

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a list of intervals:

```text
[start, end]
```

Each interval represents every point from `start` through `end`, inclusive.

For example:

```text
[1, 3]
```

covers:

```text
1, 2, 3
```

The task is to return a new list of intervals that covers exactly the same points as the input, but with all overlapping intervals combined.

For example:

```text
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
```

The first two intervals overlap:

```text
[1, 3]
[2, 6]
```

They share the region from `2` to `3`, so together they form one continuous interval:

```text
[1, 6]
```

The other intervals do not touch or overlap that region, so the final answer is:

```text
[[1, 6], [8, 10], [15, 18]]
```

So the real problem is:

> Given many possibly overlapping ranges on a line, compress them into the smallest set of non-overlapping ranges that covers the same total region.

The answer does not need to remember which original intervals created each merged interval. It only needs to preserve the covered span.

---

### 2. What Counts as Overlap?

Two intervals:

```text
[a, b]
[c, d]
```

overlap if they share at least one point.

If the intervals are already ordered by start, so `a <= c`, then overlap is determined by one comparison:

```text
c <= b
```

That means the second interval starts before the first interval has ended.

Examples:

```text
[1, 3] and [2, 6]
```

overlap because:

```text
2 <= 3
```

Together they become:

```text
[1, max(3, 6)] = [1, 6]
```

Touching endpoints also count as overlap in this problem:

```text
[1, 4] and [4, 5]
```

The point `4` belongs to both intervals, so they merge into:

```text
[1, 5]
```

That is why the comparison must be:

```text
next_start <= current_end
```

not:

```text
next_start < current_end
```

---

### 3. Start From the Brute Force Baseline

The most direct way to think about the problem is:

1. Pick an interval.
2. Search through all other intervals for anything that overlaps it.
3. Merge those intervals.
4. Repeat until no more merges are possible.

Conceptually:

```python
changed = True

while changed:
    changed = False

    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if intervals[i] overlaps intervals[j]:
                intervals[i] = merged interval
                remove intervals[j]
                changed = True
                break
```

This is correct in spirit, but it is awkward and inefficient.

The same interval may be compared many times. A merge can create a larger interval that then overlaps something that was already checked earlier. The order of operations becomes messy because modifying the list changes the future comparisons.

In the worst case, this repeated scanning can become quadratic or worse depending on how removals are handled.

The deeper question is:

> Can we arrange the intervals so each interval only needs to be considered once?

Yes. The key is sorting by start position.

---

### 4. The Key Observation: Overlap Is Local After Sorting

Intervals are ranges on a number line.

If we sort them by start coordinate, then we see them from left to right:

```text
[smallest start, ...]
[next start, ...]
[next start, ...]
...
```

Once intervals are in this order, an important fact becomes true:

> When scanning left to right, the next interval can only interact with the merged interval currently at the end of the answer.

Why?

Suppose we have already processed some intervals and produced merged output like this:

```text
[1, 6], [8, 10]
```

Now the next interval starts at `12`.

Because the input is sorted by start, every future interval starts at `12` or later. Therefore, no future interval can go back and overlap `[1, 6]`. The only possible interval that the next input interval could merge with is the last merged interval, `[8, 10]`.

This is the central simplification.

Instead of asking:

```text
Does this interval overlap any interval I have ever seen?
```

we only ask:

```text
Does this interval overlap the most recent merged interval?
```

That turns a pairwise problem into a single left-to-right scan.

---

### 5. The Sorted-Interval Invariant

After sorting, we maintain an output list called `merged`.

The invariant is:

```text
After processing the first k sorted intervals, merged is the fully merged, non-overlapping representation of exactly those k intervals.
```

This means two things:

1. `merged` covers the same points as the intervals already processed.
2. No two intervals inside `merged` overlap or touch.

The last interval in `merged` is special.

It is the only interval that might still grow, because future intervals start to its right. Earlier merged intervals are already sealed off.

For every next sorted interval `[start, end]`, compare it to:

```text
merged[-1] = [last_start, last_end]
```

There are only two cases.

#### Case 1: The next interval overlaps the last merged interval

This happens when:

```text
start <= last_end
```

Then the intervals belong to the same connected region, so we extend the last merged interval:

```python
last_end = max(last_end, end)
```

The start does not change, because `last_start <= start` due to sorting.

#### Case 2: The next interval starts after the last merged interval ends

This happens when:

```text
start > last_end
```

There is a real gap:

```text
last_end ... gap ... start
```

Because all future intervals start at `start` or later, nothing can ever fill that gap by reaching backward. The previous merged interval is final, and the new interval begins a new merged component.

So we append it:

```python
merged.append([start, end])
```

---

### 6. Detailed Algorithm

The algorithm is:

1. If the input is empty, return an empty list.
2. Sort intervals by their start coordinate.
3. Create an empty result list `merged`.
4. Scan the sorted intervals from left to right.
5. If `merged` is empty, add the current interval.
6. Otherwise compare the current interval with the last interval in `merged`.
7. If they overlap or touch, extend the last interval's end.
8. If they are disjoint, append the current interval as a new merged interval.
9. Return `merged`.

In Python-like pseudocode:

```python
def merge(intervals):
    intervals.sort(key=lambda interval: interval[0])

    merged = []

    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue

        last = merged[-1]

        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return merged
```

This code mutates the local ordering of `intervals` by sorting it. If you need to preserve the caller's original order, use:

```python
for start, end in sorted(intervals, key=lambda interval: interval[0]):
    ...
```

The LeetCode problem only cares about the returned merged intervals, so in-place sorting is commonly accepted.

---

### 7. Walk Through Example 1

Input:

```text
[[1, 3], [2, 6], [8, 10], [15, 18]]
```

It is already sorted by start.

Start with:

```text
merged = []
```

Process `[1, 3]`:

```text
merged is empty
append [1, 3]
```

Now:

```text
merged = [[1, 3]]
```

Process `[2, 6]`:

```text
last merged interval = [1, 3]
current start = 2
last end = 3
```

Since:

```text
2 <= 3
```

there is overlap. Extend the last interval:

```text
new end = max(3, 6) = 6
```

Now:

```text
merged = [[1, 6]]
```

Process `[8, 10]`:

```text
last merged interval = [1, 6]
current start = 8
last end = 6
```

Since:

```text
8 > 6
```

there is a gap. Append a new interval:

```text
merged = [[1, 6], [8, 10]]
```

Process `[15, 18]`:

```text
last merged interval = [8, 10]
current start = 15
last end = 10
```

Since:

```text
15 > 10
```

there is another gap. Append it:

```text
merged = [[1, 6], [8, 10], [15, 18]]
```

Return:

```text
[[1, 6], [8, 10], [15, 18]]
```

---

### 8. Walk Through Touching Endpoints

Input:

```text
[[1, 4], [4, 5]]
```

Sorted order is the same.

Process `[1, 4]`:

```text
merged = [[1, 4]]
```

Process `[4, 5]`:

```text
last merged interval = [1, 4]
current start = 4
last end = 4
```

The overlap check is:

```text
4 <= 4
```

This is true. The intervals touch at point `4`, so they merge:

```text
new end = max(4, 5) = 5
```

Return:

```text
[[1, 5]]
```

This example is the reason `<=` is required.

---

### 9. Walk Through Unsorted Input

Input:

```text
[[4, 7], [1, 4]]
```

If we process this order directly, we see `[4, 7]` before `[1, 4]`, which makes the left-to-right invariant false.

So we sort first:

```text
[[1, 4], [4, 7]]
```

Process `[1, 4]`:

```text
merged = [[1, 4]]
```

Process `[4, 7]`:

```text
current start = 4
last end = 4
4 <= 4
```

They touch and merge:

```text
merged = [[1, 7]]
```

Return:

```text
[[1, 7]]
```

Sorting is not just a convenience. It is what makes the local comparison with `merged[-1]` valid.

---

### 10. Correctness

We prove the algorithm returns exactly the merged representation of the input intervals.

#### Invariant

After processing the first `k` intervals in sorted order, `merged` is a sorted, non-overlapping list of intervals that covers exactly the same points as those first `k` intervals.

#### Base Case

Before processing any intervals, `merged` is empty.

It covers exactly the same points as an empty set of input intervals: no points.

So the invariant holds.

After processing the first interval, `merged` contains exactly that interval, so it still holds.

#### Inductive Step

Assume the invariant holds after processing the first `k` sorted intervals.

Now consider the next interval:

```text
[start, end]
```

Because intervals are sorted by start, this interval starts no earlier than every interval already processed.

Only the last interval in `merged` can possibly overlap it. All earlier intervals in `merged` end before the last interval begins or are separated from it by gaps, and their starts are even further left. Since the new interval starts at or after the last interval's start, it cannot skip over the last interval and overlap an earlier one without also overlapping the last one.

There are two cases.

If:

```text
start <= merged[-1][1]
```

then the new interval overlaps or touches the last merged interval. Replacing the last end with:

```text
max(merged[-1][1], end)
```

creates exactly the union of those two intervals. No coverage is lost, and no extra gap is introduced.

If:

```text
start > merged[-1][1]
```

then there is a gap between the last merged interval and the new interval. Since all future intervals start at `start` or later, no future interval can fill that gap from the left. Therefore the new interval must begin a separate merged interval, and appending it preserves sorted, non-overlapping coverage.

In both cases, after processing the next interval, `merged` covers exactly the first `k + 1` intervals and remains sorted and non-overlapping.

By induction, the invariant holds after all intervals are processed.

At that point, `merged` covers exactly the same points as the entire input and contains no overlapping intervals, so it is the required answer.

---

### 11. Complexity

Let `n` be the number of intervals.

Sorting costs:

```text
O(n log n)
```

The scan visits each interval once:

```text
O(n)
```

So the total time complexity is:

```text
O(n log n)
```

The output list can contain up to `n` intervals, so output space is:

```text
O(n)
```

If we do not count the returned output, the extra working space is usually:

```text
O(1)
```

when sorting in place, though the language's sorting implementation may use additional internal space.

---

### 12. Common Pitfalls

#### Using `<` instead of `<=`

For this problem, touching intervals merge.

```text
[1, 4] and [4, 5] -> [1, 5]
```

So the condition must be:

```python
if start <= merged[-1][1]:
```

Using `<` would incorrectly return:

```text
[[1, 4], [4, 5]]
```

#### Forgetting to sort first

Without sorting, comparing only against the last merged interval is not reliable.

Input such as:

```text
[[4, 7], [1, 4]]
```

requires sorting before the scan can safely merge it into:

```text
[[1, 7]]
```

#### Replacing the end instead of taking the maximum

When intervals overlap, the merged interval's end is:

```python
max(last_end, end)
```

not simply `end`.

For example:

```text
[1, 10] and [2, 3]
```

merge into:

```text
[1, 10]
```

If you set the end to `3`, you would lose coverage from `4` through `10`.

#### Appending every interval before deciding whether it merges

The output list should contain final merged components, not every original interval.

Each new interval should either extend `merged[-1]` or start a new component. It should not do both.

#### Accidentally sharing mutable input intervals

If you append the original interval objects and then mutate them, you may also mutate lists from the input. LeetCode accepts this, but production code often copies intervals before modifying them:

```python
merged.append([start, end])
```

instead of:

```python
merged.append(interval)
```

---

### 13. First-Principles Summary

The problem is about compressing coverage on a number line.

The brute-force approach repeatedly asks whether arbitrary pairs of intervals overlap. That is expensive because without order, every interval might need to be compared to many others.

Sorting by start coordinate creates structure. Once intervals are sorted, we scan from left to right, and all processed intervals are behind us. The only interval that can still be extended is the last merged interval in the answer.

That gives the core invariant:

```text
merged is the complete merged result for everything processed so far.
```

For each next interval, there are only two possibilities:

```text
It overlaps merged[-1] -> extend merged[-1].
It starts after merged[-1] -> append a new interval.
```

This is why the solution is short in code but important in reasoning. The sort turns global overlap relationships into local, final decisions.

## Implementation
See `solutions/intervals/p056_merge_intervals.py`.

## Tests
See `tests/intervals/test_p056_merge_intervals.py`.

## Examples

### Example 1
- Input: `{'intervals': [[1, 3], [2, 6], [8, 10], [15, 18]]}`
- Output: `[[1, 6], [8, 10], [15, 18]]`

### Example 2
- Input: `{'intervals': [[1, 4], [4, 5]]}`
- Output: `[[1, 5]]`

### Example 3
- Input: `{'intervals': [[4, 7], [1, 4]]}`
- Output: `[[1, 7]]`

## Follow-up Practice
- Draw intervals on a number line before writing code.
- Sort the intervals and trace only the last merged interval.
- Test overlapping, touching, nested, disjoint, and unsorted intervals.
- Explain why earlier merged intervals can never change once a gap appears.
