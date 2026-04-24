# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When the score depends on two ends and a shrinking distance, evaluate the widest remaining pair and move the side that limits the score.

## Why Two Pointers Fits

The container area is `width * min(left_height, right_height)`. Starting with the widest possible width gives the best distance. Once a pair is evaluated, moving the taller side cannot improve the limiting height, because the shorter side still caps the area and the width gets smaller. Only moving the shorter side can possibly find a taller limiting wall.

## Recommended Approach

1. Put `left` at the first bar and `right` at the last bar.
2. Compute the area between them.
3. Update the best area seen so far.
4. Move the pointer at the shorter bar inward.
5. If both bars have equal height, moving either side is safe; this implementation moves `right`.
6. Continue until the pointers meet.

## Alternative Approaches

The brute-force approach checks every pair of bars, which takes `O(n^2)` time. A dynamic-programming table is unnecessary because the key decision is local and monotonic: once width shrinks, the only hope is to improve the limiting height. The two-pointer method captures that directly.

## Correctness Sketch

Consider a pair `(left, right)` where `height[left] <= height[right]`. Any container using `left` with a smaller right index has less width and a height no greater than `height[left]`, so it cannot beat the current pair. Therefore, after evaluating `(left, right)`, it is safe to discard `left`. The symmetric argument applies when the right bar is shorter. The algorithm evaluates one representative before discarding each impossible boundary, so the maximum area is never skipped.

## Trace

For `[1, 8, 6, 2, 5, 4, 8, 3, 7]`:

| Left height | Right height | Width | Area | Move |
| --- | --- | --- | --- | --- |
| `1` | `7` | `8` | `8` | Move left, shorter side |
| `8` | `7` | `7` | `49` | Move right, shorter side |
| `8` | `3` | `6` | `18` | Move right |
| `8` | `8` | `5` | `40` | Move right on tie |

The best area remains `49`.

## Complexity

- Time: `O(n)` because one pointer moves on every iteration.
- Space: `O(1)` because only a few integers are stored.

## Common Pitfalls

- Moving the taller side and losing the dominance argument.
- Forgetting that width shrinks as pointers move inward.
- Trying to sort heights, which destroys the original positions and widths.
- Using the taller height instead of the shorter height in the area formula.

## Implementation Notes

See `solutions/two_pointers/p011_container_with_most_water.py`. The implementation keeps the area formula explicit: `width * min(height[left], height[right])`.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`. The tests cover official examples, two-bar input, monotonic height arrays, and equal-height arrays.

## Interview Script

"I start with the widest container. The shorter wall limits the current area, and keeping that wall while reducing width cannot improve the answer. So after checking a pair, I move the shorter side inward and keep the best area seen."

## Review Questions

1. Why is moving the shorter side the only move that can improve the answer?
2. Why would sorting the heights break the problem?
3. What does the width represent in the area formula?
4. Why is the brute-force solution `O(n^2)`?

## Follow-up Practice

- Trapping Rain Water, which also reasons about boundary heights.
- Maximize a score formed by two boundary values and distance.
- Prove dominance arguments for other two-pointer problems.
