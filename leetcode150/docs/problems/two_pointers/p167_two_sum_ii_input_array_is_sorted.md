# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## Core Pattern

When a sorted array asks for a pair with a target sum, compare the smallest and largest remaining candidates. Move the side that is guaranteed not to help.

## Why Two Pointers Fits

The array is sorted in nondecreasing order. If `numbers[left] + numbers[right]` is too small, every pair using `numbers[left]` with an index smaller than `right` is also too small, so `left` can move right. If the sum is too large, every pair using `numbers[right]` with an index larger than `left` is also too large, so `right` can move left.

## Recommended Approach

1. Set `left = 0` and `right = len(numbers) - 1`.
2. Compute `current_sum = numbers[left] + numbers[right]`.
3. If the sum equals `target`, return the 1-indexed pair `[left + 1, right + 1]`.
4. If the sum is smaller than `target`, increment `left` to increase the sum.
5. If the sum is larger than `target`, decrement `right` to decrease the sum.
6. Continue until the answer is found.

## Alternative Approaches

A hash map can solve the unsorted version in linear time, but it uses extra space and ignores the sorted input. Binary-searching the complement for every index gives `O(n log n)` time. The two-pointer method uses the sorted order directly and achieves `O(n)` time with `O(1)` space.

## Correctness Sketch

At each step, the answer must lie within the current `[left, right]` window. If the current sum is too small, `numbers[left]` cannot pair with any remaining value at or left of `right` to reach the target, because those values are no larger than `numbers[right]`. Therefore discarding `left` is safe. The too-large case symmetrically proves that discarding `right` is safe. Since each move discards only impossible candidates and the problem guarantees one answer, the algorithm eventually returns the correct pair.

## Trace

For `numbers = [2, 7, 11, 15]`, `target = 9`:

| Left | Right | Sum | Action |
| --- | --- | --- | --- |
| `2` at index 1 | `15` at index 4 | `17` | Too large, move `right` left |
| `2` at index 1 | `11` at index 3 | `13` | Too large, move `right` left |
| `2` at index 1 | `7` at index 2 | `9` | Return `[1, 2]` |

## Complexity

- Time: `O(n)` because each pointer moves inward at most `n` times total.
- Space: `O(1)` because no auxiliary data structure is needed.

## Common Pitfalls

- Returning zero-based indices instead of 1-indexed positions.
- Using a hash map and missing the constant-space advantage of sorted input.
- Moving both pointers when the sum is not equal to the target.
- Forgetting that duplicate values can be the correct pair.

## Implementation Notes

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`. The implementation raises `ValueError` only as defensive code; valid LeetCode inputs contain exactly one solution.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`. The tests cover official examples, negative values, minimal input, and duplicate values forming the unique answer.

## Interview Script

"Because the array is sorted, I start with the smallest and largest values. If their sum is too small, the smaller value cannot work with anything else, so I move left. If the sum is too large, the larger value cannot work with anything else, so I move right. That discards impossible pairs until the guaranteed answer is found."

## Review Questions

1. Why does sorted order make it safe to move only one pointer?
2. Why must the returned indices be shifted by one?
3. How does this differ from the unsorted Two Sum problem?
4. Why do duplicates not require special handling here?

## Follow-up Practice

- Solve the unsorted Two Sum problem with a hash map.
- Count pairs with sum less than a target in a sorted array.
- Extend the idea to 3Sum by fixing one number and scanning the rest.
