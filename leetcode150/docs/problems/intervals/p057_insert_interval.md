# 57. Insert Interval

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/insert-interval/
- Official Group: Intervals
- Pattern Group: Intervals
- Patterns: intervals

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a list of intervals:

```text
intervals = [[start1, end1], [start2, end2], ...]
```

and one extra interval:

```text
newInterval = [newStart, newEnd]
```

The input intervals have two important properties already:

1. They are sorted by start time.
2. They do not overlap each other.

You need to insert `newInterval` into that list and return a new sorted, non-overlapping list.

If `newInterval` overlaps one or more existing intervals, all of those intervals must be merged into one larger interval.

For example:

```text
intervals   = [[1, 3], [6, 9]]
newInterval = [2, 5]
```

The new interval touches the first interval because:

```text
[1, 3] and [2, 5] overlap
```

Together they cover:

```text
[1, 5]
```

The interval `[6, 9]` is separate, so the answer is:

```text
[[1, 5], [6, 9]]
```

The real problem is:

> Place one new range into an already clean timeline, merging exactly the intervals whose covered positions collide with it.

This is not asking us to solve a general unsorted interval problem. The input is already organized, and the solution should use that organization.

---

### 2. Start From the Brute Force Idea

A straightforward baseline is:

1. Append `newInterval` to `intervals`.
2. Sort all intervals by their start point.
3. Run the standard merge-intervals algorithm.

Conceptually:

```python
all_intervals = intervals + [newInterval]
all_intervals.sort(key=lambda interval: interval[0])

merged = []

for start, end in all_intervals:
    if not merged or merged[-1][1] < start:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)

return merged
```

This is correct because, after sorting, overlapping intervals become adjacent.

But it ignores a valuable input guarantee:

```text
intervals is already sorted and already non-overlapping
```

Sorting everything again costs `O(n log n)`, even though only one interval is new. We can do better by scanning once.

---

### 3. The Key Observation

Because the existing intervals are sorted and non-overlapping, every existing interval belongs to exactly one of three groups relative to `newInterval`:

```text
completely before newInterval
overlapping with newInterval
completely after newInterval
```

For an interval `[start, end]`:

It is completely before `newInterval` if:

```text
end < newStart
```

It overlaps `newInterval` if:

```text
start <= newEnd and end >= newStart
```

It is completely after `newInterval` if:

```text
start > newEnd
```

Since the intervals are sorted, these groups appear in this exact order:

```text
before intervals -> overlapping intervals -> after intervals
```

There cannot be a "before" interval after an overlapping interval, and there cannot be an overlapping interval after an "after" interval. The sorted order rules that out.

That means we can solve the problem with one left-to-right pass:

1. Copy all intervals that end before the new interval begins.
2. Merge every interval that overlaps the new interval.
3. Copy all remaining intervals.

---

### 4. The Sorted-Interval Invariant

The input gives us a strong invariant before we do anything:

```text
For every adjacent pair intervals[i] and intervals[i + 1]:
intervals[i][0] <= intervals[i + 1][0]
intervals[i][1] < intervals[i + 1][0]
```

In words:

```text
starts are sorted, and each interval ends before the next interval starts
```

The output must satisfy the same invariant.

The algorithm preserves it by only appending intervals when it is safe:

1. Intervals before `newInterval` are appended unchanged.
   - They end before `newInterval` starts.
   - They are already sorted among themselves.

2. The merged version of `newInterval` is appended once.
   - It begins after every copied-before interval.
   - It absorbs every interval that overlaps it.

3. Intervals after the merged interval are appended unchanged.
   - The first after interval starts after the merged interval ends.
   - They are already sorted among themselves.

The important invariant while scanning is:

```text
newInterval always represents the full merged coverage of the original new interval
plus every overlapping interval seen so far.
```

When we see an overlapping interval `[start, end]`, we update:

```text
newStart = min(newStart, start)
newEnd   = max(newEnd, end)
```

Because the old `newInterval` and `[start, end]` overlap, their union is still one continuous interval.

---

### 5. Detailed Algorithm

Let:

```text
newStart = newInterval[0]
newEnd   = newInterval[1]
i        = 0
result   = []
```

#### Phase 1: Copy intervals completely before the new interval

While the current interval ends before `newInterval` begins:

```text
intervals[i][1] < newStart
```

it cannot overlap `newInterval`, and no later merge can change it. Append it directly to `result` and move forward.

Why is it final?

Because later intervals start even later, and this interval is already safely to the left of the new interval.

#### Phase 2: Merge intervals that overlap the new interval

Now the current interval is not strictly before `newInterval` anymore.

As long as the current interval starts before or at the current merged end:

```text
intervals[i][0] <= newEnd
```

it overlaps the current merged interval and must be absorbed.

Update the merged interval:

```text
newStart = min(newStart, intervals[i][0])
newEnd   = max(newEnd, intervals[i][1])
```

Then advance `i`.

Notice why `newEnd` may grow. Suppose:

```text
newInterval = [4, 8]
current     = [8, 10]
```

They overlap at `8`, so the merged interval becomes `[4, 10]`.

That larger end may now overlap still later intervals. The algorithm handles this naturally because the loop condition uses the updated `newEnd`.

#### Phase 3: Append the merged new interval

After Phase 2, there are no more intervals that overlap the merged interval.

Append:

```text
[newStart, newEnd]
```

This is the only new/changed interval in the output.

#### Phase 4: Copy the remaining intervals

Every remaining interval starts after `newEnd`, so it is completely to the right of the merged interval.

Append them unchanged.

---

### 6. Pseudocode

```python
def insert(intervals, newInterval):
    result = []
    i = 0
    n = len(intervals)

    new_start = newInterval[0]
    new_end = newInterval[1]

    # 1. Everything strictly before the new interval is already final.
    while i < n and intervals[i][1] < new_start:
        result.append(intervals[i])
        i += 1

    # 2. Every interval that overlaps extends the merged new interval.
    while i < n and intervals[i][0] <= new_end:
        new_start = min(new_start, intervals[i][0])
        new_end = max(new_end, intervals[i][1])
        i += 1

    # 3. Insert the merged interval exactly once.
    result.append([new_start, new_end])

    # 4. Everything after it is already final.
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

A compact equivalent implementation can also append `newInterval` as a mutable current interval and update its endpoints in place. The idea is the same: copy left side, absorb overlaps, copy right side.

---

### 7. Detailed Example Walkthrough

Use the second example:

```text
intervals   = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
newInterval = [4, 8]
```

Initialize:

```text
result   = []
newStart = 4
newEnd   = 8
i        = 0
```

#### Phase 1: intervals before `[4, 8]`

Current interval:

```text
[1, 2]
```

Check whether it is completely before the new interval:

```text
2 < 4  -> true
```

So append it:

```text
result = [[1, 2]]
i = 1
```

Current interval:

```text
[3, 5]
```

Check:

```text
5 < 4  -> false
```

So `[3, 5]` is not completely before the new interval. It may overlap, so Phase 1 stops.

#### Phase 2: merge overlaps

Current interval:

```text
[3, 5]
```

Check whether it overlaps the current merged interval `[4, 8]`:

```text
3 <= 8  -> true
```

Merge:

```text
newStart = min(4, 3) = 3
newEnd   = max(8, 5) = 8
```

Now the merged interval is:

```text
[3, 8]
```

Move to the next interval:

```text
i = 2
```

Current interval:

```text
[6, 7]
```

Check:

```text
6 <= 8  -> true
```

Merge:

```text
newStart = min(3, 6) = 3
newEnd   = max(8, 7) = 8
```

Merged interval remains:

```text
[3, 8]
```

Move forward:

```text
i = 3
```

Current interval:

```text
[8, 10]
```

Check:

```text
8 <= 8  -> true
```

This is important: intervals that touch at an endpoint count as overlapping for this problem. `[3, 8]` and `[8, 10]` should become `[3, 10]`.

Merge:

```text
newStart = min(3, 8) = 3
newEnd   = max(8, 10) = 10
```

Now:

```text
merged interval = [3, 10]
i = 4
```

Current interval:

```text
[12, 16]
```

Check:

```text
12 <= 10  -> false
```

So Phase 2 stops. `[12, 16]` is completely after the merged interval.

#### Phase 3: append the merged interval

```text
result = [[1, 2], [3, 10]]
```

#### Phase 4: append the rest

Append `[12, 16]`:

```text
result = [[1, 2], [3, 10], [12, 16]]
```

That is the final answer.

---

### 8. Correctness

We prove that the algorithm returns exactly the sorted, non-overlapping interval list produced by inserting `newInterval`.

#### Lemma 1: Every interval appended in Phase 1 belongs unchanged in the output.

Phase 1 appends an interval only when:

```text
interval.end < newStart
```

So the interval is strictly before the original new interval. Since the input intervals are sorted and non-overlapping, all intervals appended before it are also before it, and no later interval can make this interval overlap `newInterval`. Therefore it must appear unchanged in the output.

#### Lemma 2: Phase 2 merges exactly the intervals that overlap the inserted interval's merged component.

Phase 2 continues while:

```text
interval.start <= currentMergedEnd
```

At that point, Phase 1 has already removed every interval whose end is strictly before the merged interval's start. Therefore the current interval is not completely before the merged interval. If its start is also not after the merged end, the two intervals overlap or touch, so they must be merged.

Each merge updates the current merged interval to the union of itself and the overlapping interval. Because overlapping intervals form one continuous covered range, this update preserves exactly the total coverage seen so far.

When Phase 2 stops, either there are no intervals left, or the next interval satisfies:

```text
interval.start > currentMergedEnd
```

Since later intervals start even later, no remaining interval can overlap the merged interval. Thus Phase 2 merged all and only the required intervals.

#### Lemma 3: The merged interval is placed in the only valid position.

All Phase 1 intervals end before the merged interval begins, and all remaining intervals start after the merged interval ends. Therefore the merged interval belongs after the Phase 1 intervals and before the remaining intervals. Appending it at that point preserves sorted order and non-overlap.

#### Lemma 4: Every interval appended after the merged interval belongs unchanged in the output.

After Phase 2 stops, the current interval, if any, starts after the merged interval ends. Because the input is sorted and non-overlapping, every later interval starts even later and remains non-overlapping with the merged interval and with the other remaining intervals. Therefore all remaining intervals should be copied unchanged.

#### Theorem: The algorithm returns the correct result.

By Lemma 1, all intervals before the inserted interval are copied correctly. By Lemma 2, all intervals that must merge with the inserted interval are merged into exactly one interval. By Lemma 3, that merged interval is inserted in the correct position. By Lemma 4, all intervals after it are copied correctly. These groups cover every input interval plus `newInterval`, so the returned list is exactly the required sorted, non-overlapping result.

---

### 9. Complexity

Let `n` be the number of existing intervals.

Each interval is inspected at most once:

```text
Phase 1 scans some prefix.
Phase 2 scans the overlapping middle.
Phase 4 scans the remaining suffix.
```

So the time complexity is:

```text
O(n)
```

The output may contain up to `n + 1` intervals, so the output space is:

```text
O(n)
```

Ignoring the returned list itself, the algorithm uses only a few variables:

```text
O(1) extra space
```

This improves on the brute-force append-sort-merge baseline, which costs `O(n log n)` time because of sorting.

---

### 10. Common Pitfalls

#### Pitfall 1: Sorting again unnecessarily

Sorting works, but it misses the point of the problem's input guarantee. The existing intervals are already sorted and disjoint, so one scan is enough.

#### Pitfall 2: Using the wrong comparison for overlaps

For closed intervals, touching endpoints overlap:

```text
[1, 3] and [3, 5] -> [1, 5]
```

So the merge condition should allow equality:

```text
interval.start <= newEnd
```

If you use `< newEnd`, you will fail to merge endpoint-touching intervals.

#### Pitfall 3: Forgetting that `newEnd` can grow

After merging an interval, the merged interval may extend farther right. That can cause it to overlap later intervals that did not overlap the original `newInterval`.

Example:

```text
intervals   = [[1, 2], [5, 7], [8, 10]]
newInterval = [3, 6]
```

Merging `[3, 6]` with `[5, 7]` gives `[3, 7]`. If a later interval started at `7`, it would now also need to merge. Always compare against the updated end.

#### Pitfall 4: Appending the new interval too early or too often

The inserted interval should be appended exactly once, after all overlaps have been absorbed. Appending it before merging, or appending it inside the merge loop, creates duplicates or partial intervals.

#### Pitfall 5: Mishandling empty input

If `intervals` is empty, Phase 1 and Phase 2 do nothing, then the algorithm appends `newInterval` and returns:

```text
[newInterval]
```

No special case is required if the loop structure is written cleanly.

#### Pitfall 6: Accidentally mutating caller-owned input

Some implementations update `newInterval[0]` and `newInterval[1]` directly. That is often accepted on LeetCode, but using local variables like `newStart` and `newEnd` avoids surprising mutation.

---

### 11. First-Principles Summary

The problem becomes simple once you stop thinking of insertion as "try every possible position" and instead think of the number line.

The existing intervals already partition the line into sorted, non-overlapping covered pieces. The new interval can only do three things as we scan from left to right:

```text
come after an interval,
merge with an interval,
or come before an interval.
```

Because the intervals are sorted, those cases happen in one irreversible order:

```text
before -> overlap -> after
```

That ordering is the whole solution.

Copy the safe left side, absorb the overlapping middle into one expanded interval, then copy the safe right side. The sorted/non-overlapping invariant is preserved because each append happens only when its relative position is already final.

## Implementation
See `solutions/intervals/p057_insert_interval.py`.

## Tests
See `tests/intervals/test_p057_insert_interval.py`.

## Examples

### Example 1
- Input: `{'intervals': [[1, 3], [6, 9]], 'newInterval': [2, 5]}`
- Output: `[[1, 5], [6, 9]]`

### Example 2
- Input: `{'intervals': [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], 'newInterval': [4, 8]}`
- Output: `[[1, 2], [3, 10], [12, 16]]`

## Follow-up Practice
- Trace `intervals = []`, `newInterval = [2, 3]`.
- Trace inserting before every interval, such as `newInterval = [0, 0]` into `[[1, 2], [3, 4]]`.
- Trace inserting after every interval, such as `newInterval = [5, 6]` into `[[1, 2], [3, 4]]`.
- Trace an interval that absorbs everything, such as `newInterval = [0, 99]`.
- Explain why the three groups must appear as `before`, then `overlap`, then `after`.
