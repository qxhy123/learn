# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Tags: two-pointers, greedy, array

## Core Pattern

When a score is the product of width and a limiting boundary, start with the widest span and move the boundary that currently limits the score. The key idea is dominance: once one boundary is proven too short to help, keeping it while shrinking the width cannot produce a better answer.

## Why Two Pointers Fits

The container area is `width * min(left_height, right_height)`. As the pointers move inward, width only decreases, so the only way to offset that loss is to find a taller limiting wall. The shorter wall is the bottleneck, because the taller wall does not control the current area. Moving the taller wall cannot improve the minimum height, so it cannot help enough to compensate for the smaller width.

Sorting is not valid here because the horizontal positions are part of the problem. If you reorder the bars, you change every width and destroy the original container geometry.

## Recommended Approach

1. Place `left` at the first bar and `right` at the last bar.
2. Compute `width = right - left` and `area = width * min(height[left], height[right])`.
3. Update `best_area` if the current area is larger.
4. Move the pointer at the shorter bar inward.
5. If the heights are equal, either move is safe; choose one consistently.
6. Repeat until `left` and `right` meet.

## Alternative Approaches

The brute-force solution checks every pair of bars and takes `O(n^2)` time. That is easy to reason about, but it wastes the monotonic structure in the problem. There is no useful sorting trick here because the original positions are part of the answer. The two-pointer method is the cleanest way to apply a dominance argument directly.

## Correctness Sketch

Consider a pair `(left, right)` where `height[left] <= height[right]`. The current area is limited by `height[left]`. Any container that keeps `left` and moves `right` inward will have smaller width and a height no greater than `height[left]`, so it cannot beat the current pair. Therefore `left` can be discarded after the current area is evaluated. The symmetric argument applies when the right wall is shorter. Because each step removes only dominated boundaries after recording the area for the current widest remaining pair, the maximum area cannot be skipped.

## Trace

For `[1, 8, 6, 2, 5, 4, 8, 3, 7]`:

| Left height | Right height | Width | Area | Best so far | Move |
| --- | --- | --- | --- | --- | --- |
| `1` | `7` | `8` | `8` | `8` | Move left, shorter side |
| `8` | `7` | `7` | `49` | `49` | Move right, shorter side |
| `8` | `3` | `6` | `18` | `49` | Move right |
| `8` | `8` | `5` | `40` | `49` | Move right on tie |

The best area stays `49` because later pairs have less width and do not find a taller limiting wall than the one already recorded.

## Complexity

- Time: `O(n)` because one pointer moves on every iteration.
- Space: `O(1)` because the algorithm stores only indices and a few scalar values.

## Common Pitfalls

- Moving the taller side and breaking the dominance argument.
- Using `max(height[left], height[right])` instead of the shorter wall in the area formula.
- Forgetting that width decreases every time the pointers move inward.
- Sorting the heights and accidentally changing the problem.
- Assuming the local best-looking pair is always the global best without checking the full dominance logic.

## Implementation Notes

See `solutions/two_pointers/p011_container_with_most_water.py`. The implementation computes `width * min(height[left], height[right])` explicitly and moves the shorter pointer inward on each step. On equal heights, the current code moves `right`, but moving `left` would also preserve correctness.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`. The tests cover the official examples, a two-bar input, monotonic increasing and decreasing heights, and equal-height arrays.

## Interview Script

"I start with the widest container because width is part of the score. The shorter wall limits the current area, so keeping it while shrinking width cannot improve the answer. After measuring each pair, I move the shorter side inward and keep the maximum area I have seen."

## Review Questions

1. Why is the shorter side the only move that can possibly improve the answer?
2. Why does sorting the heights break the problem?
3. What exactly does the width term represent?
4. Why is the brute-force method `O(n^2)`?
5. What should happen when both walls have the same height?

## Follow-up Practice

- Trapping Rain Water, which also depends on boundary heights.
- Any problem where the score is a product of distance and a limiting value.
- Proving dominance arguments in other greedy two-pointer problems.
