# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When two boundaries define a score and the distance between them shrinks as pointers move, evaluate the widest remaining pair first and discard the boundary that provably limits all future pairs using it.

## Why Two Pointers Fits

The area for two bars is `width * min(height[left], height[right])`. The width is largest when the pointers start at the two ends. After any pointer move, width decreases. Therefore, a future improvement can only come from increasing the limiting height.

If the left bar is shorter, keeping it while moving the right bar inward cannot help: the width gets smaller and the height is still capped by the same left bar. The only move with potential is to discard the shorter bar and search for a taller one. This dominance argument is exactly what the two-pointer method captures.

## Recommended Approach

1. Set `left = 0`, `right = len(height) - 1`, and `best_area = 0`.
2. While `left < right`, compute `width = right - left`.
3. Compute the limiting height as `min(height[left], height[right])`.
4. Update `best_area` with `width * limiting_height`.
5. Move the pointer at the shorter bar inward.
6. If the heights are equal, moving either side is safe; the implementation moves `right`.
7. Return `best_area` after the pointers meet.

## Alternative Approaches

A brute-force solution checks every pair of bars and takes `O(n^2)` time, which is too slow for `n` up to `10^5`. Sorting the heights is invalid because the original indices determine width. Dynamic programming does not add value because the local dominance proof tells us exactly which boundary can be discarded.

## Correctness Sketch

Assume `height[left] <= height[right]`. For any index `k` with `left < k < right`, the container formed by `(left, k)` has smaller width than `(left, right)` and limiting height at most `height[left]`. Its area therefore cannot exceed the area already evaluated for `(left, right)`. So no optimal solution is lost by discarding `left`. The argument is symmetric when `height[right] < height[left]`. Since each step records the current area before discarding a boundary and discards only boundaries that cannot improve the answer, the maximum area is preserved in `best_area`.

## Trace

For `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`:

| `left` | `right` | Width | Limiting height | Area | Move |
| --- | --- | --- | --- | --- | --- |
| `0 / 1` | `8 / 7` | `8` | `1` | `8` | move `left` |
| `1 / 8` | `8 / 7` | `7` | `7` | `49` | move `right` |
| `1 / 8` | `7 / 3` | `6` | `3` | `18` | move `right` |
| `1 / 8` | `6 / 8` | `5` | `8` | `40` | tie, move `right` |

The best recorded area is `49`.

## Complexity

- Time: `O(n)` because each iteration moves exactly one pointer inward.
- Space: `O(1)` because only pointer indices and the best area are stored.

## Common Pitfalls

- Moving the taller side without a proof.
- Using the taller bar in the area formula instead of the shorter bar.
- Sorting bars and losing the original widths.
- Forgetting to evaluate the current area before moving a pointer.
- Assuming the tallest two bars always form the best container; width matters as much as height.

## Implementation Notes

See `solutions/two_pointers/p011_container_with_most_water.py`. The code explicitly computes `width`, `current_height`, and `best_area`, then moves the shorter side according to the dominance proof.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`. The tests cover official examples, the minimum two-bar input, increasing and decreasing height arrays, and equal-height arrays.

## Interview Script

"I start with the widest possible container. The shorter wall limits the area. If I keep that shorter wall and move the taller wall inward, width decreases and the limiting height cannot improve, so that cannot beat the current pair. Therefore I record the area and move the shorter side."

## Review Questions

1. Why does the shorter wall limit the current area?
2. What exactly is proven before a boundary is discarded?
3. Why are the tallest two bars not always the answer?
4. Why is sorting invalid even though this is a two-pointer problem?
5. What makes this proof different from the sorted-sum proof in Two Sum II?

## Follow-up Practice

- Trapping Rain Water.
- Problems where two endpoints and distance define a score.
- Writing dominance proofs for pointer movement decisions.
