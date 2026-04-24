# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Tags: two-pointers, sorted-array, sum

## Core Pattern

In a sorted array, compare the smallest and largest remaining candidates for a target sum. Move only the side that is guaranteed not to help, because the sorted order turns every pointer move into a safe elimination.

## Why Two Pointers Fits

The input is sorted in nondecreasing order, so the sum changes predictably when a pointer moves. If the current sum is too small, increasing the left pointer is the only move that can make the sum larger. If the current sum is too large, decreasing the right pointer is the only move that can make the sum smaller. That monotonic behavior is exactly what two pointers exploit.

## Recommended Approach

1. Set `left = 0` and `right = len(numbers) - 1`.
2. Compute `current_sum = numbers[left] + numbers[right]`.
3. If `current_sum == target`, return `[left + 1, right + 1]` because LeetCode expects 1-indexed positions.
4. If `current_sum < target`, move `left += 1` to increase the sum.
5. If `current_sum > target`, move `right -= 1` to decrease the sum.
6. Continue until the unique solution is found.

The problem guarantees exactly one solution, so the search should end with a direct return rather than a collection of candidates.

## Alternative Approaches

A hash map solves the unsorted Two Sum problem in linear time, but it ignores the sorted order and uses extra memory. Another option is to binary-search the complement for each index, which gives `O(n log n)` time. The two-pointer version is the cleanest answer here because the sorted input already gives you the elimination rule for free.

## Correctness Sketch

Maintain this invariant: if the answer exists, it lies somewhere inside the current `[left, right]` window. When the sum is too small, `numbers[left]` cannot pair with any element at or to the left of `right` to reach the target, because all of those elements are `<= numbers[right]`, so every such pair is also too small. Discarding `left` is therefore safe. The too-large case is symmetric: `numbers[right]` cannot pair with any element at or to the right of `left` to reach the target, so discarding `right` is safe. Because each move removes only impossible candidates and the contract promises exactly one answer, the algorithm must eventually return the correct 1-indexed pair.

## Trace

For `numbers = [2, 7, 11, 15]` and `target = 9`:

| Left index | Right index | Pair | Sum | Action |
| --- | --- | --- | --- | --- |
| `1` | `4` | `2 + 15` | `17` | Too large, move `right` left |
| `1` | `3` | `2 + 11` | `13` | Too large, move `right` left |
| `1` | `2` | `2 + 7` | `9` | Return `[1, 2]` |

The elimination story is the same for negative values or duplicate values, because the sorted order still guarantees monotonic sums.

## Complexity

- Time: `O(n)` because each pointer moves inward at most `n` times total.
- Space: `O(1)` because the algorithm only stores pointer indices and the running sum.

## Common Pitfalls

- Returning zero-based indices instead of 1-indexed positions.
- Moving both pointers at once instead of eliminating one impossible side at a time.
- Falling back to a hash map and missing the sorted-order advantage.
- Overthinking duplicate values even though the contract guarantees a unique solution.
- Forgetting that the “exactly one solution” guarantee is part of the reasoning.

## Implementation Notes

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`. The implementation returns the 1-indexed answer as soon as it finds the matching pair. The defensive `ValueError` is only there to guard against invalid input outside the LeetCode contract.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`. The tests cover the official examples, negative values, the minimal two-element input, and duplicate values that still produce the unique valid pair.

## Interview Script

"Because the array is sorted, I can eliminate one side at a time. If the sum is too small, the left value is too small to work with anything else on the right, so I move left. If the sum is too large, the right value is too large to work with anything else on the left, so I move right. When the sum matches, I return the 1-indexed pair immediately."

## Review Questions

1. Why does sorted order make it safe to move only one pointer?
2. Why does the answer have to be converted to 1-indexed positions?
3. How is this different from the unsorted Two Sum problem?
4. Why do duplicate values not need special handling here?
5. How does the “exactly one solution” contract simplify the loop?

## Follow-up Practice

- Solve the unsorted Two Sum problem with a hash map.
- Count pairs with sum less than a target in a sorted array.
- Use the same left/right elimination idea as the inner loop of 3Sum.
