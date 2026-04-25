# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## First-Principles Explanation

### What The Problem Is Asking

You are given an array `height`, where `height[i]` is the height of a vertical line drawn at horizontal position `i`.

Pick exactly two lines. Together with the x-axis, those two lines form a container. The container's water level cannot rise above the shorter of the two selected lines, because water would spill over that side.

For two chosen indices `i` and `j`, where `i < j`:

```text
width  = j - i
height = min(height[i], height[j])
area   = width * height
```

The task is to return the largest possible area over all pairs of indices.

Two facts matter more than any category label:

- A wide pair can still be bad if one wall is short.
- A tall pair can still be bad if the two walls are close together.

So the problem is a tradeoff between distance and the shorter boundary.

### Brute-Force Baseline

The direct solution is to try every pair of lines:

```text
best = 0
for left from 0 to n - 1:
    for right from left + 1 to n - 1:
        width = right - left
        level = min(height[left], height[right])
        best = max(best, width * level)
return best
```

This is correct because it evaluates the area for every possible container. There is no missed candidate.

The cost is the problem: there are `n * (n - 1) / 2` pairs, so the time complexity is `O(n^2)`. For large inputs, that is too slow.

To improve, we need a way to discard many pairs without individually checking them.

### Key Observation

Suppose the current pair is `(left, right)`.

The area is:

```text
(right - left) * min(height[left], height[right])
```

Now assume `height[left] <= height[right]`. The left wall is the limiting wall. The current container can hold water only up to `height[left]`, no matter how tall the right wall is.

Consider every other pair that still uses this same `left` index but moves the right index inward:

```text
(left, right - 1)
(left, right - 2)
...
(left, left + 1)
```

Every one of those pairs has smaller width than `(left, right)`. Also, every one has effective height at most `height[left]`, because the left wall is still present and still caps the water level.

Therefore, none of those narrower containers using the same `left` wall can beat the area already computed for `(left, right)`.

That means after evaluating `(left, right)`, if `height[left] <= height[right]`, the index `left` can be discarded completely. It cannot be part of a better future answer.

The symmetric argument applies when `height[right] <= height[left]`: after evaluating the current pair, all future pairs using the same `right` wall and a larger `left` boundary have smaller width and are capped by `height[right]`, so `right` can be discarded.

This is the entire reason the two-pointer algorithm works.

### Two-Pointer Invariant

Maintain two pointers:

- `left`: the leftmost boundary not yet discarded.
- `right`: the rightmost boundary not yet discarded.
- `best_area`: the largest area among all pairs evaluated so far.

The invariant is:

> Before each iteration, every pair that has already been discarded has been proven unable to beat `best_area`, and any pair that might still improve the answer lies within the active range `[left, right]`.

At each step, evaluate the widest remaining pair `(left, right)`. Then discard the shorter side, because keeping that shorter side while shrinking width cannot produce a better area.

This invariant is stronger than merely saying “move two pointers inward.” It explains why the move is safe.

### Discard Proof

Assume `height[left] <= height[right]`.

After evaluating `(left, right)`, the recorded candidate area is:

```text
current = (right - left) * height[left]
```

For any index `k` with `left < k < right`, the area of pair `(left, k)` is:

```text
(k - left) * min(height[left], height[k])
```

Since `k < right`:

```text
k - left < right - left
```

And since the left wall is still part of the pair:

```text
min(height[left], height[k]) <= height[left]
```

So:

```text
(k - left) * min(height[left], height[k])
<=(k - left) * height[left]
<(right - left) * height[left]
= current
```

Every future pair using index `left` is no better than the current pair, which has already been considered. Thus `left` can be safely moved to `left + 1`.

When `height[right] < height[left]`, the same proof shows that `right` can be safely moved to `right - 1`.

When the two heights are equal, either side may be moved. The implementation moves `right` in that case because the condition is `if height[left] < height[right]`, otherwise move `right`. This is safe because both sides are equally limiting; discarding either one follows the same argument.

### Detailed Algorithm

1. Start with the widest possible container: `left = 0`, `right = len(height) - 1`.
2. Initialize `best_area = 0`.
3. While `left < right`:
   - Compute the width: `right - left`.
   - Compute the limiting height: `min(height[left], height[right])`.
   - Update `best_area` with this container's area.
   - If the left wall is shorter, move `left` one step right.
   - Otherwise, move `right` one step left.
4. Return `best_area`.

The algorithm starts with maximum width and gradually gives up width only when it has proven that one boundary cannot help anymore. The only possible way to compensate for the shrinking width is to find a taller limiting wall, so the pointer on the shorter wall is the only one worth moving.

### Pseudocode

```text
function maxArea(height):
    left = 0
    right = length(height) - 1
    best_area = 0

    while left < right:
        width = right - left
        level = min(height[left], height[right])
        best_area = max(best_area, width * level)

        if height[left] < height[right]:
            left = left + 1
        else:
            right = right - 1

    return best_area
```

This matches the Python implementation linked below.

### Detailed Example Walkthrough

Use the main example:

```text
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
index    0  1  2  3  4  5  6  7  8
```

Start with the widest pair.

| Step | `left` | `right` | Heights | Width | Area | Best | Move |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8 | `1, 7` | 8 | 8 | 8 | Move `left`; height `1` is limiting. |
| 2 | 1 | 8 | `8, 7` | 7 | 49 | 49 | Move `right`; height `7` is limiting. |
| 3 | 1 | 7 | `8, 3` | 6 | 18 | 49 | Move `right`; height `3` is limiting. |
| 4 | 1 | 6 | `8, 8` | 5 | 40 | 49 | Equal heights; move `right` by implementation choice. |
| 5 | 1 | 5 | `8, 4` | 4 | 16 | 49 | Move `right`; height `4` is limiting. |
| 6 | 1 | 4 | `8, 5` | 3 | 15 | 49 | Move `right`; height `5` is limiting. |
| 7 | 1 | 3 | `8, 2` | 2 | 4 | 49 | Move `right`; height `2` is limiting. |
| 8 | 1 | 2 | `8, 6` | 1 | 6 | 49 | Move `right`; height `6` is limiting. |

The pointers meet, so the search stops. The answer is `49`, produced by indices `1` and `8`:

```text
width = 8 - 1 = 7
level = min(8, 7) = 7
area  = 7 * 7 = 49
```

Notice the important moment at step 2. The algorithm does not know yet that `49` is globally optimal. It only knows that after checking `(1, 8)`, index `8` can be discarded because the right wall of height `7` limits the container, and moving inward would only reduce width unless a different right boundary is used. The invariant protects the search until all useful candidates have either been checked or safely discarded.

### Correctness

We prove that the algorithm returns the maximum possible container area.

#### Lemma 1: Moving the shorter pointer does not discard an optimal unexamined pair.

Consider an iteration with pointers `left` and `right`.

If `height[left] <= height[right]`, then for every `k` where `left < k < right`, pair `(left, k)` has smaller width than `(left, right)` and limiting height at most `height[left]`. Therefore its area is at most the area already computed for `(left, right)`. So no unexamined pair using `left` can be better than what has already been recorded, and discarding `left` is safe.

If `height[right] < height[left]`, the symmetric argument shows that no unexamined pair using `right` can be better than the current evaluated pair, so discarding `right` is safe.

#### Lemma 2: Every discarded pair is either evaluated or proven unable to improve the answer.

The only discarded pairs are those involving the pointer moved in an iteration. By Lemma 1, all such pairs are bounded by the current pair's area. Since the current pair is evaluated before the move and `best_area` is updated, those discarded pairs cannot exceed `best_area`.

#### Theorem: `best_area` equals the maximum possible area.

The algorithm stops when `left >= right`, meaning no pair remains in the active range. By Lemma 2, every pair outside the active range was either evaluated directly or proven unable to beat an evaluated pair included in `best_area`. Therefore no possible pair has area greater than `best_area`. Since `best_area` is always the area of some evaluated valid container, it is exactly the maximum possible area.

### Complexity

- Time: `O(n)`. Each iteration moves exactly one pointer inward, so there are at most `n - 1` iterations.
- Space: `O(1)`. The algorithm stores only two pointers and the best area.

### Common Pitfalls

- Moving the taller wall. If the shorter wall stays, the limiting height cannot increase, and the width definitely shrinks.
- Forgetting that the area uses the shorter height, not the taller height or an average of the two.
- Using `right - left + 1` for width. The lines are at indices, so the horizontal distance is `right - left`.
- Stopping at `left <= right`. A container needs two distinct lines, so the loop condition should be `left < right`.
- Sorting the heights. Sorting destroys the original horizontal positions, and width depends on those positions.
- Assuming equal heights require moving both pointers. Moving either one is safe; moving one at a time keeps the proof simple and still stays linear.

### First-Principles Summary

The maximum container is determined by two numbers: the distance between the chosen lines and the shorter of their heights. Starting from the widest possible pair gives the largest available width. Once that pair is evaluated, the shorter side can be discarded because any container reusing it with a smaller width cannot have a taller limiting height than that shorter side. Repeating this argument shrinks the search space from both ends while preserving every candidate that could still matter.

The algorithm is not a template trick. It is a sequence of safe eliminations: evaluate the current widest remaining pair, prove the shorter boundary cannot help again, discard it, and keep the best area seen.

## Implementation

See `solutions/two_pointers/p011_container_with_most_water.py`.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`.

## Examples

- Official example: `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]` returns `49`.
- Two bars only: `height = [1, 1]` returns `1`.
- Monotonic heights: `height = [1, 2, 3, 4, 5]` returns `6`, and `height = [5, 4, 3, 2, 1]` returns `6`.
- Equal heights: `height = [5, 5, 5, 5]` returns `15`.
- See `tests/two_pointers/test_p011_container_with_most_water.py` for executable examples and edge cases.

## Follow-up Practice

- For each pointer move, write down which group of pairs was discarded.
- Prove why moving the taller wall cannot improve the area while the shorter wall remains.
- Trace the algorithm on strictly increasing, strictly decreasing, and all-equal arrays.
