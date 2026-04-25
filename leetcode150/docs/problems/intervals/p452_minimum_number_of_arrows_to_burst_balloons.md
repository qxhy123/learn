# 452. Minimum Number of Arrows to Burst Balloons

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
- Official Group: Intervals
- Pattern Group: Intervals
- Patterns: intervals

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Each balloon is represented by an interval on the x-axis:

```text
[start, end]
```

A balloon occupies every x-position from `start` through `end`, inclusive.

An arrow is shot vertically upward at exactly one x-position. If we shoot an arrow at position `x`, it bursts every balloon whose interval contains `x`:

```text
start <= x <= end
```

The task is to find the minimum number of arrow positions needed so that every interval contains at least one chosen arrow position.

So the problem is not really about simulating arrows flying upward. It is this interval-covering question:

> Given many closed intervals on a line, choose as few points as possible so that every interval contains at least one chosen point.

For example:

```text
points = [[10, 16], [2, 8], [1, 6], [7, 12]]
```

One optimal choice is:

```text
shoot at x = 6
shoot at x = 12
```

The arrow at `6` bursts:

```text
[1, 6]
[2, 8]
```

The arrow at `12` bursts:

```text
[7, 12]
[10, 16]
```

Every balloon is burst, so the answer is `2`.

---

### 2. Start From the Brute Force Idea

A direct way to think about the problem is:

1. Pick some arrow positions.
2. Check whether every balloon contains at least one of those positions.
3. Try to use fewer arrows.

But the possible arrow positions are not just the integer coordinates shown in the input. Coordinates can be very large, and conceptually an arrow could be shot at any point along the x-axis.

A slightly smarter brute force observation is:

> There is always an optimal solution that shoots arrows at balloon endpoints.

If an arrow is inside a group of overlapping intervals, sliding it right until it reaches the smallest ending point among those intervals does not make it leave any of those intervals. So we only need to consider endpoint-like choices.

Even then, a brute force subset search is far too expensive. With `n` balloons, there can be many endpoint choices, and trying all subsets is exponential.

The brute force mindset is still useful because it reveals what an arrow must do:

```text
One arrow is useful only for a group of balloons that all share at least one common x-position.
```

If several intervals overlap at a common point, one arrow can burst all of them. If the next interval starts after every possible shared point has already ended, then no arrow for the old group can help with the new interval.

That is the first hint that we should process intervals in an order where this decision becomes final.

---

### 3. The Key Observation

For one arrow to burst several balloons, their intervals must have a non-empty intersection.

For example:

```text
[1, 6]
[2, 8]
[7, 12]
```

The first two intervals overlap:

```text
[1, 6] ∩ [2, 8] = [2, 6]
```

So one arrow can burst both, at any position from `2` to `6`.

But adding `[7, 12]` destroys the shared intersection:

```text
[2, 6] ∩ [7, 12] = empty
```

No single x-position can be inside all three intervals. Therefore the third balloon needs a different arrow from the one used for the first group.

The problem becomes:

```text
Partition the intervals into as few groups as possible,
where each group has a common intersection point.
```

Instead of explicitly building groups, we can greedily choose arrow positions.

The safest place to shoot an arrow for the current uncovered balloon is at its right endpoint. Shooting at the right endpoint still bursts that balloon, and it leaves the arrow as far right as possible, maximizing the chance that it also lies inside future balloons.

That is the central greedy idea:

> Sort balloons by their ending coordinate, then shoot at the earliest ending balloon's end whenever the current arrow cannot burst the next balloon.

---

### 4. Why Sort by End?

Suppose the unburst balloon with the smallest end is:

```text
[a, b]
```

Any valid solution must use some arrow inside `[a, b]`, because this balloon has to be burst.

Among all positions inside `[a, b]`, choosing `b` is never worse than choosing an earlier point:

```text
b is as far right as possible while still bursting [a, b]
```

Future balloons, after sorting by end, may start before or after `b`. A later arrow position has the best chance of still being inside them.

For example, if a future balloon is:

```text
[c, d]
```

and it can be burst by some arrow inside `[a, b]`, then its start must satisfy:

```text
c <= chosen_arrow
```

Choosing the largest possible arrow position `b` can only help satisfy `c <= arrow`; it cannot make the arrow too far right for this future balloon because `d >= b` when intervals are sorted by end and `[a, b]` is the earliest-ending active balloon.

So once we commit to bursting the earliest-ending uncovered balloon, shooting at its end is a safe local decision.

---

### 5. The Greedy Interval Invariant

After sorting by `end`, scan from left to right and maintain:

```text
arrows     = number of arrows already shot
arrow_pos  = x-position of the most recent arrow
```

The invariant is:

```text
All balloons processed so far are burst using arrows arrows,
and arrow_pos is the rightmost useful position chosen for the current group.
```

More concretely, after we shoot an arrow at the end of some interval:

```text
arrow_pos = that interval's end
```

Every later processed balloon falls into one of two cases.

Case 1: The balloon starts at or before `arrow_pos`.

```text
start <= arrow_pos
```

Because the intervals are sorted by end, this balloon's end is at least `arrow_pos`. Therefore:

```text
start <= arrow_pos <= end
```

The current arrow bursts it. No new arrow is needed.

Case 2: The balloon starts after `arrow_pos`.

```text
start > arrow_pos
```

Then the current arrow is strictly to the left of this balloon and cannot burst it. Since all future balloons end no earlier than this one, no adjustment to the old arrow can cover both the previous earliest-ending group and this balloon. A new arrow is unavoidable.

So the greedy choice is final:

```text
if start <= arrow_pos:
    current arrow covers this balloon
else:
    shoot a new arrow at this balloon's end
```

The inclusive comparison matters because intervals are closed. If one balloon ends at `2` and another starts at `2`, an arrow at `2` bursts both.

---

### 6. Detailed Algorithm

If there are no balloons, the answer is `0`.

Otherwise:

1. Sort `points` by each interval's ending coordinate.
2. Shoot the first arrow at the end of the first interval.
3. Set `arrows = 1`.
4. For each remaining interval `[start, end]`:
   - If `start <= arrow_pos`, the existing arrow is inside this interval, so do nothing.
   - If `start > arrow_pos`, the existing arrow cannot burst this interval, so:
     - increment `arrows`
     - set `arrow_pos = end`
5. Return `arrows`.

In Python-like pseudocode:

```python
def findMinArrowShots(points):
    if not points:
        return 0

    points.sort(key=lambda interval: interval[1])

    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows
```

The implementation can also initialize `arrows = 0` and `arrow_pos = -infinity`, then handle every interval uniformly. The invariant is the same: whenever the current interval is not covered by the latest arrow, create a new arrow at that interval's end.

---

### 7. Detailed Walkthrough

Use the first example:

```text
points = [[10, 16], [2, 8], [1, 6], [7, 12]]
```

Sort by end:

```text
[1, 6]
[2, 8]
[7, 12]
[10, 16]
```

Start with the earliest-ending balloon:

```text
current interval = [1, 6]
shoot arrow at x = 6
arrows = 1
```

Now process `[2, 8]`:

```text
start = 2
arrow_pos = 6
2 <= 6
```

The arrow at `6` lies inside `[2, 8]`, so it bursts this balloon too.

State remains:

```text
arrow_pos = 6
arrows = 1
```

Now process `[7, 12]`:

```text
start = 7
arrow_pos = 6
7 > 6
```

The arrow at `6` is left of this balloon. It cannot burst `[7, 12]`.

Because `[1, 6]` ended at `6`, no arrow that bursts `[1, 6]` can also burst a balloon that starts at `7`. A second arrow is necessary.

Shoot the new arrow at this balloon's end:

```text
arrow_pos = 12
arrows = 2
```

Now process `[10, 16]`:

```text
start = 10
arrow_pos = 12
10 <= 12
```

The arrow at `12` lies inside `[10, 16]`, so it bursts this balloon too.

Final result:

```text
arrows = 2
```

So the answer is `2`.

---

### 8. Touching Intervals Example

Consider:

```text
points = [[1, 2], [2, 3], [3, 4], [4, 5]]
```

Sorted by end, the order is already:

```text
[1, 2]
[2, 3]
[3, 4]
[4, 5]
```

Shoot the first arrow at `2`:

```text
[1, 2] is burst
[2, 3] is also burst because 2 is inside [2, 3]
```

Then `[3, 4]` starts after `2`, so we need a new arrow at `4`:

```text
[3, 4] is burst
[4, 5] is also burst because 4 is inside [4, 5]
```

The answer is `2`.

This example is important because it shows why the condition for needing a new arrow is:

```text
start > arrow_pos
```

not:

```text
start >= arrow_pos
```

When `start == arrow_pos`, the current arrow still hits the balloon.

---

### 9. Correctness Argument

We prove that the greedy algorithm returns the minimum possible number of arrows.

First, after sorting by end, consider the first unburst balloon during the scan:

```text
[start, end]
```

This balloon must be burst by any valid solution, so any valid solution must place some arrow inside this interval.

The greedy algorithm places an arrow at exactly:

```text
end
```

This choice is safe. Suppose another valid solution places the arrow for this balloon at some position `x` where:

```text
start <= x <= end
```

Move that arrow from `x` to `end`.

This still bursts the current balloon. For any later balloon that was also burst by `x`, its end is at least `end`, because intervals are processed in nondecreasing end order. If it contained `x` and starts at or before `x`, moving the arrow right to `end` preserves coverage whenever that later balloon starts at or before `end`; those are exactly the balloons the greedy arrow covers. Moving to `end` therefore does not reduce the set of future intervals that can be grouped with this earliest-ending balloon in an optimal way.

So there exists an optimal solution whose first arrow is placed where the greedy algorithm places it.

After that arrow is placed, every interval containing `arrow_pos` is already burst and requires no additional arrow. Every interval whose start is greater than `arrow_pos` cannot be burst by this arrow. Since the earliest-ending balloon in the previous group ends at `arrow_pos`, no single arrow can cover both that previous group and an interval starting after `arrow_pos`.

Therefore, when the algorithm encounters an interval with:

```text
start > arrow_pos
```

adding a new arrow is not just convenient; it is necessary.

The same argument repeats for the remaining unburst intervals. Each greedy arrow can be assumed to match an arrow in some optimal solution, and each time the algorithm starts a new group, any solution must also use another arrow.

Thus the greedy algorithm uses no more arrows than an optimal solution and never uses fewer arrows than necessary. It returns the minimum number of arrows.

---

### 10. Complexity

Let `n` be the number of balloons.

Sorting dominates the running time:

```text
O(n log n)
```

The scan after sorting is linear:

```text
O(n)
```

So the total time complexity is:

```text
O(n log n)
```

The extra space complexity is:

```text
O(1)
```

if sorting is treated as in-place and we ignore the language/runtime's internal sorting stack. In Python, `list.sort` may use additional internal memory, but the algorithm itself only maintains a counter and one arrow position.

---

### 11. Common Pitfalls

- Using `start >= arrow_pos` to decide that a new arrow is needed. This is wrong because balloon intervals are closed, so `start == arrow_pos` means the arrow hits the balloon.
- Sorting by start and then greedily shooting at starts. The safe greedy choice is tied to the earliest ending uncovered balloon, because its end is the latest position that is guaranteed to still burst it.
- Updating `arrow_pos` when an interval is already covered. If `start <= arrow_pos`, keep the existing arrow. Moving it to the current interval's end can break coverage for earlier balloons in the same group.
- Counting overlapping intervals instead of arrows. The answer is the number of disjoint arrow groups, not the number of balloons in any group.
- Forgetting the empty input case if using manual initialization from `points[0]`.
- Treating this like interval merging output. We do not need to build merged intervals; we only need the count of arrow positions.

---

### 12. First-Principles Summary

The problem asks for the smallest set of points that intersects every balloon interval.

The earliest-ending uncovered balloon creates a deadline:

```text
an arrow for that balloon must be placed no later than its end
```

Placing the arrow exactly at that end is optimal because it satisfies the current deadline while leaving the arrow as far right as possible for future balloons.

After sorting by end, the scan becomes final and local:

```text
If the current balloon starts before or at the latest arrow, it is already burst.
If it starts after the latest arrow, no previous arrow can burst it, so a new arrow is required.
```

That invariant turns the global minimum problem into one pass after sorting.

## Implementation
See `solutions/intervals/p452_minimum_number_of_arrows_to_burst_balloons.py`.

## Tests
See `tests/intervals/test_p452_minimum_number_of_arrows_to_burst_balloons.py`.

## Examples

### Example 1
- Input: `{'points': [[10, 16], [2, 8], [1, 6], [7, 12]]}`
- Output: `2`

### Example 2
- Input: `{'points': [[1, 2], [3, 4], [5, 6], [7, 8]]}`
- Output: `4`

### Example 3
- Input: `{'points': [[1, 2], [2, 3], [3, 4], [4, 5]]}`
- Output: `2`

## Follow-up Practice

- Explain why shooting at the earliest ending uncovered balloon's end is safe.
- Trace a case with touching intervals, such as `[1, 2]` and `[2, 3]`.
- Trace a case with no overlaps, where every balloon needs its own arrow.
- Describe why an already-covered interval must not move the current arrow position.
