# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## Core Pattern

Use opposite-end pointers on sorted data when comparing a pair against a target. If the pair is too small, move the smaller side up; if the pair is too large, move the larger side down.

## Why Two Pointers Fits

The array is sorted in nondecreasing order, and the official input guarantees exactly one answer. Sorting gives a monotonic relationship between pointer movement and the pair sum:

- moving `left` right can only increase or preserve the left value;
- moving `right` left can only decrease or preserve the right value.

Therefore, every sum comparison gives information about a whole set of impossible pairs, not just the current pair.

## Recommended Approach

1. Initialize `left = 0` and `right = len(numbers) - 1`.
2. Compute `current_sum = numbers[left] + numbers[right]`.
3. If `current_sum == target`, return `[left + 1, right + 1]` because the problem uses 1-indexed positions.
4. If `current_sum < target`, increment `left`; the current left value is too small to work with any remaining right value.
5. If `current_sum > target`, decrement `right`; the current right value is too large to work with any remaining left value.
6. Continue until the guaranteed pair is found.

## Alternative Approaches

The original unsorted Two Sum problem is usually solved with a hash map in `O(n)` time and `O(n)` space. Here, a hash map would work but would ignore the sorted-order advantage. Another option is to fix each index and binary-search its complement, which uses little space but takes `O(n log n)` time. The two-pointer solution is both linear and constant-space.

## Correctness Sketch

Maintain this invariant: if the answer has not been returned yet, its indices lie inside the current `[left, right]` window.

If `numbers[left] + numbers[right] < target`, then pairing `numbers[left]` with any index between `left + 1` and `right` cannot reach the target because all those partners are at most `numbers[right]`. Thus no valid answer uses `left`, so incrementing `left` preserves the invariant. If the sum is too large, then pairing `numbers[right]` with any index between `left` and `right - 1` is also too large or no smaller than needed, so no valid answer uses `right`; decrementing `right` preserves the invariant. Since the input has exactly one solution and each step removes only impossible candidates, the algorithm returns the correct 1-indexed pair.

## Trace

For `numbers = [2, 7, 11, 15]`, `target = 9`:

| `left` index/value | `right` index/value | Sum | Decision |
| --- | --- | --- | --- |
| `0 / 2` | `3 / 15` | `17` | too large, move `right` left |
| `0 / 2` | `2 / 11` | `13` | too large, move `right` left |
| `0 / 2` | `1 / 7` | `9` | return `[1, 2]` |

The output uses 1-indexing, so zero-based indices `0` and `1` become `1` and `2`.

## Complexity

- Time: `O(n)` because one pointer moves on every iteration.
- Space: `O(1)` because no auxiliary table is required.

## Common Pitfalls

- Returning zero-based indices instead of 1-indexed indices.
- Adding duplicate-skipping logic from 3Sum; duplicates are valid values here and may form the answer.
- Testing with arrays that contain multiple valid pairs, which violates the problem's exactly-one-solution contract.
- Moving both pointers after a nonmatching sum.
- Using a hash map without explaining why the sorted input makes it unnecessary.

## Implementation Notes

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`. The final `ValueError` is defensive documentation of the input contract; valid LeetCode test cases always return inside the loop.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`. The tests cover official examples, negative numbers, the minimum two-element case, and duplicate values that form the unique valid answer.

## Interview Script

"Because the array is sorted, I can start with the smallest and largest remaining values. If their sum is too small, the smallest value cannot pair with anything else in the window, so I move left. If their sum is too large, the largest value cannot pair with anything else, so I move right. When the sum matches, I return the two positions using 1-indexing."

## Review Questions

1. What set of pairs is eliminated when `current_sum < target`?
2. Why does this problem not need duplicate skipping?
3. How does the exactly-one-solution guarantee simplify the implementation?
4. Why is binary search per index slower than two pointers here?
5. What changes if the array is not sorted?

## Follow-up Practice

- Two Sum on an unsorted array.
- Count pairs with sum below a target in a sorted array.
- 3Sum, which uses this two-pointer scan as an inner step.
