# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## Core Pattern

In a sorted array, start with the smallest and largest remaining candidates. If their sum is too small, discard the smaller candidate; if their sum is too large, discard the larger candidate.

## Why Two Pointers Fits

Sorted order gives the algorithm a monotonic direction. With `left < right`, increasing `left` can only keep or increase the chosen left value, and decreasing `right` can only keep or decrease the chosen right value. Therefore, each comparison tells us which side cannot participate in the target pair.

The problem also guarantees exactly one valid answer, so the algorithm does not need to collect every pair or handle ambiguity. It only needs to find the one pair while preserving the required 1-indexed return format.

## Recommended Approach

1. Set `left = 0` and `right = len(numbers) - 1`.
2. Compute `current_sum = numbers[left] + numbers[right]`.
3. If `current_sum == target`, return `[left + 1, right + 1]`.
4. If `current_sum < target`, increment `left` because the smaller value is too small to work with any remaining right candidate.
5. If `current_sum > target`, decrement `right` because the larger value is too large to work with any remaining left candidate.
6. Continue until the guaranteed answer is found.

## Alternative Approaches

The unsorted Two Sum problem is usually solved with a hash map, but that spends `O(n)` extra space and does not use the sorted input. Another option is to fix one index and binary-search for its complement, which uses constant extra space but takes `O(n log n)` time. The two-pointer method is both linear and constant-space because it uses sorted order at every step.

## Correctness Sketch

Maintain this invariant: the unique valid pair, if not already returned, lies inside the current `[left, right]` window. If `numbers[left] + numbers[right] < target`, then `numbers[left]` paired with any index at most `right` is also too small, so no valid answer uses `left`; moving `left` preserves the invariant. If the sum is too large, then `numbers[right]` paired with any index at least `left` is also too large, so no valid answer uses `right`; moving `right` preserves the invariant. Since each step discards only impossible candidates and the input has exactly one solution, the algorithm must return that solution.

## Trace

For `numbers = [2, 7, 11, 15]`, `target = 9`:

| `left` value | `right` value | Sum | Decision |
| --- | --- | --- | --- |
| `2` | `15` | `17` | Too large, move `right` left |
| `2` | `11` | `13` | Too large, move `right` left |
| `2` | `7` | `9` | Return `[1, 2]` |

The returned indices are 1-indexed, so zero-based `(0, 1)` becomes `[1, 2]`.

## Complexity

- Time: `O(n)` because each iteration moves exactly one pointer inward.
- Space: `O(1)` because the method stores only two indices and the current sum.

## Common Pitfalls

- Returning zero-based indices.
- Using a hash map and missing the sorted-array optimization.
- Moving both pointers when the sum is too small or too large.
- Testing with inputs that have multiple valid answers even though the problem guarantees exactly one.
- Adding duplicate-skipping logic; duplicates are allowed and may be the answer.

## Implementation Notes

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`. The defensive `ValueError` is unreachable for valid LeetCode inputs, but it makes the function explicit about its required contract.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`. The tests cover official examples, negative numbers, two-element input, and duplicate values under the exactly-one-solution contract.

## Interview Script

"Because the array is sorted, I compare the smallest and largest remaining numbers. If the sum is too small, the smallest number cannot work with any remaining partner, so I move left. If the sum is too large, the largest number cannot work, so I move right. When the sum matches, I return the two positions using 1-indexing."

## Review Questions

1. What sorted-order fact justifies moving `left` when the sum is too small?
2. Why is no duplicate-skipping step needed?
3. Why is the answer returned with `+1` on each index?
4. How does the problem's exactly-one-solution guarantee affect testing?
5. When would a hash-map solution be more appropriate?

## Follow-up Practice

- Original Two Sum on an unsorted array.
- Count pairs with sum less than a target in a sorted array.
- 3Sum, which fixes one value and then uses this pattern on the suffix.
