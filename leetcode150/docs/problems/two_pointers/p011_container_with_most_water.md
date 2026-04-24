# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When a score is determined by two boundaries and the distance between them, start with the widest boundary pair and move the boundary that currently limits the score.

## Why Two Pointers Fits

The area between two bars is `width * min(height[left], height[right])`. Starting at both ends maximizes width. After evaluating a pair, width will only shrink, so the only way to improve the area is to find a taller limiting bar. Moving the taller bar cannot increase the limiting height while the shorter bar remains in place; moving the shorter bar is the only move with upside.

This is a dominance-style two-pointer problem: each move discards a boundary after proving it cannot produce a better answer with any narrower partner.

## Recommended Approach

1. Initialize `left = 0`, `right = len(height) - 1`, and `best_area = 0`.
2. Compute the current width as `right - left`.
3. Compute the current limiting height as `min(height[left], height[right])`.
4. Update `best_area` with the current area.
5. Move the pointer at the shorter bar inward.
6. If both bars are equal, moving either one is safe; the implementation moves `right`.
7. Repeat until the pointers meet, then return `best_area`.

## Alternative Approaches

The brute-force solution checks every pair and takes `O(n^2)` time. Sorting is invalid because positions determine width; reordering bars changes the problem. A stack or dynamic-programming table is unnecessary because the dominance argument gives a direct local decision at each step.

## Correctness Sketch

Assume `height[left] <= height[right]`. For any `k` with `left < k < right`, the container `(left, k)` has smaller width than `(left, right)` and limiting height at most `height[left]`. Therefore it cannot exceed the area already computed for `(left, right)`. So after evaluating `(left, right)`, no optimal solution uses `left`, and moving `left` is safe. The same reasoning applies symmetrically when `height[right] < height[left]`. Since every discarded boundary is proven unable to improve the best area, the algorithm never discards an optimal answer before considering an area at least as large.

## Trace

For `[1, 8, 6, 2, 5, 4, 8, 3, 7]`:

| Left index/value | Right index/value | Width | Area | Move |
| --- | --- | --- | --- | --- |
| `0 / 1` | `8 / 7` | `8` | `8` | Move left because `1` limits area |
| `1 / 8` | `8 / 7` | `7` | `49` | Move right because `7` limits area |
| `1 / 8` | `7 / 3` | `6` | `18` | Move right |
| `1 / 8` | `6 / 8` | `5` | `40` | Tie, move right |

The best value found is `49`.

## Complexity

- Time: `O(n)` because exactly one pointer moves on each iteration.
- Space: `O(1)` because the algorithm stores only pointer positions and the best area.

## Common Pitfalls

- Moving the taller bar and losing the proof that the discarded side is impossible.
- Using the taller height in the area formula instead of the shorter height.
- Sorting the heights and destroying width information.
- Forgetting that width shrinks every time a pointer moves.
- Trying to prove the move by intuition instead of the dominance argument.

## Implementation Notes

See `solutions/two_pointers/p011_container_with_most_water.py`. The implementation keeps the limiting-height calculation explicit, which makes the pointer move easy to audit.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`. The tests cover official examples, the minimal two-bar case, monotonic arrays, and equal-height bars.

## Interview Script

"I start with the widest possible container. The shorter wall limits the current area; if I keep it and move the taller wall inward, width decreases and the limiting height cannot improve. So after recording the area, I move the shorter side and keep the best area seen."

## Review Questions

1. Why does the shorter side determine the current area's maximum possible height?
2. Why is sorting invalid for this problem?
3. What exactly is proven when we discard a boundary?
4. Why is moving either pointer safe when the heights are equal?
5. How is this dominance argument different from the sorted-sum argument in Two Sum II?

## Follow-up Practice

- Trapping Rain Water.
- Maximize a score involving two endpoints and distance.
- Practice writing dominance proofs for pointer movement.
