# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## First-Principles Explanation

### What The Problem Asks

You are given an integer array `numbers` that is already sorted in non-decreasing order, and an integer `target`.

You must find two distinct elements whose values add up to `target`, then return their positions using **1-based indexing**.

That means if the answer uses `numbers[0]` and `numbers[1]`, the returned answer is `[1, 2]`, not `[0, 1]`.

The problem also promises that:

- There is exactly one valid answer.
- You may not use the same element twice.
- The input array is sorted before the algorithm begins.

For example:

```text
numbers = [2, 7, 11, 15]
target = 9
```

The values `2 + 7 = 9`, so the answer is `[1, 2]`.

The key challenge is not merely finding a pair. The key challenge is using the sorted order so the search is linear and uses constant extra space.

### Brute-Force Baseline

Start from the most direct idea: try every possible pair.

```text
for i from 0 to n - 1:
    for j from i + 1 to n - 1:
        if numbers[i] + numbers[j] == target:
            return [i + 1, j + 1]
```

This is correct because it checks every pair of distinct indices. If the promised answer exists, brute force will eventually inspect it.

But it is wasteful. An array of length `n` has roughly `n * (n - 1) / 2` pairs, so this takes `O(n^2)` time.

The sorted order is not being used at all. If the array were unsorted, this brute-force algorithm would behave the same way. That is a sign that the solution has not yet exploited the special structure of this problem.

### Key Observation

Because `numbers` is sorted, moving right makes values stay the same or become larger, and moving left makes values stay the same or become smaller.

So for any current pair:

```text
left value  = numbers[left]
right value = numbers[right]
current sum = numbers[left] + numbers[right]
```

There are only three cases:

1. `current_sum == target`: the pair is the answer.
2. `current_sum < target`: the sum is too small.
3. `current_sum > target`: the sum is too large.

The important part is what each inequality proves.

If `numbers[left] + numbers[right] < target`, then `numbers[left]` is too small even when paired with the largest value still available, `numbers[right]`. Pairing `numbers[left]` with anything between `left + 1` and `right - 1` can only make the sum smaller or equal, because those values are no larger than `numbers[right]`.

So once the sum is too small, the current `left` index cannot be part of the answer. It is safe to discard it by moving `left` one step right.

If `numbers[left] + numbers[right] > target`, then `numbers[right]` is too large even when paired with the smallest value still available, `numbers[left]`. Pairing `numbers[right]` with anything between `left + 1` and `right - 1` can only make the sum larger or equal, because those values are no smaller than `numbers[left]`.

So once the sum is too large, the current `right` index cannot be part of the answer. It is safe to discard it by moving `right` one step left.

This is the whole reason two pointers work here: each comparison eliminates an entire row or column of impossible pairs, not just one pair.

### Sorted Two-Pointer Invariant

Use two pointers:

- `left` starts at the first index, `0`.
- `right` starts at the last index, `len(numbers) - 1`.

At every step, maintain this invariant:

```text
If the answer has not been found yet, then the answer is still inside the active range [left, right].
```

More specifically, every index outside that range has already been proven impossible:

- Any index before `left` was discarded because its value was too small to reach the target with the largest remaining partner.
- Any index after `right` was discarded because its value was too large to reach the target with the smallest remaining partner.

The algorithm only moves a pointer after proving that the pointer's current index cannot appear in the answer. Therefore the real answer is never skipped.

Because the problem guarantees exactly one answer, the loop will find it before the pointers cross.

### Detailed Algorithm

1. Set `left = 0`.
2. Set `right = len(numbers) - 1`.
3. While `left < right`:
   - Compute `current_sum = numbers[left] + numbers[right]`.
   - If `current_sum == target`, return `[left + 1, right + 1]`.
   - If `current_sum < target`, move `left += 1` because the left value is too small.
   - Otherwise, move `right -= 1` because the right value is too large.
4. If the loop exits, the input violated the problem guarantee that exactly one solution exists.

The condition `left < right` matters because the two chosen elements must be distinct. When `left == right`, the algorithm would be trying to use the same element twice, which is not allowed.

### Detailed Example Walkthrough

Consider:

```text
numbers = [2, 7, 11, 15]
target = 9
```

Start with the widest possible search range:

```text
left = 0  -> numbers[left] = 2
right = 3 -> numbers[right] = 15
sum = 2 + 15 = 17
```

`17` is greater than `9`, so the sum is too large.

Because the array is sorted, `15` is the largest value. If `15` is already too large with the smallest available value `2`, then `15` will also be too large with `7` or `11`. Therefore index `3` cannot be part of the answer.

Move `right` left:

```text
left = 0  -> numbers[left] = 2
right = 2 -> numbers[right] = 11
sum = 2 + 11 = 13
```

`13` is still greater than `9`, so `11` is too large with the smallest available value. Discard index `2`.

Move `right` left again:

```text
left = 0  -> numbers[left] = 2
right = 1 -> numbers[right] = 7
sum = 2 + 7 = 9
```

The sum equals the target. The indices are `0` and `1`, but the problem asks for 1-based positions, so return:

```text
[1, 2]
```

Now consider a case where the left pointer moves:

```text
numbers = [-5, -2, 1, 4, 9]
target = 7
```

Initial pair:

```text
left = 0  -> -5
right = 4 -> 9
sum = 4
```

`4` is less than `7`, so the sum is too small. Since `-5 + 9` is too small even with the largest available partner, `-5` cannot form the target with any remaining value. Move `left` right.

Next pair:

```text
left = 1  -> -2
right = 4 -> 9
sum = 7
```

The target is found. Return `[2, 5]` because the required output is 1-based.

### Code

The implementation in this repository uses the same pointer logic:

```python
class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            current_sum = numbers[left] + numbers[right]
            if current_sum == target:
                return [left + 1, right + 1]
            if current_sum < target:
                left += 1
            else:
                right -= 1

        raise ValueError("Input must contain exactly one solution")
```

The final `ValueError` is defensive. Under the LeetCode problem constraints, the loop should always return from inside the loop because exactly one solution is guaranteed.

### Correctness

We prove that the algorithm returns the correct 1-based indices.

First, the algorithm starts with `left = 0` and `right = len(numbers) - 1`, so every possible answer pair is inside the active range.

Now consider one loop iteration.

If `numbers[left] + numbers[right] == target`, the algorithm returns the two positions. The indices are distinct because the loop only runs while `left < right`, and the returned positions are `left + 1` and `right + 1`, which matches the problem's 1-based indexing requirement.

If `numbers[left] + numbers[right] < target`, then for every index `k` with `left < k <= right`, we have `numbers[k] <= numbers[right]` because the array is sorted. Therefore:

```text
numbers[left] + numbers[k] <= numbers[left] + numbers[right] < target
```

So `numbers[left]` cannot pair with any remaining value to reach `target`. Moving `left` right discards no valid answer.

If `numbers[left] + numbers[right] > target`, then for every index `k` with `left <= k < right`, we have `numbers[k] >= numbers[left]` because the array is sorted. Therefore:

```text
numbers[k] + numbers[right] >= numbers[left] + numbers[right] > target
```

So `numbers[right]` cannot pair with any remaining value to reach `target`. Moving `right` left discards no valid answer.

Thus every pointer move preserves the invariant that the real answer, if not already returned, remains inside the active range. The active range shrinks on every iteration, and the problem guarantees that exactly one answer exists. Therefore the algorithm must eventually examine and return that answer.

### Complexity

- Time: `O(n)`, because each iteration moves exactly one pointer, and each pointer moves across the array at most once.
- Space: `O(1)`, because the algorithm uses only a few variables besides the returned list.

This improves on the brute-force `O(n^2)` baseline by using sorted order to discard many impossible pairs at once.

### Common Pitfalls

- Returning 0-based indices. LeetCode 167 requires 1-based positions, so return `[left + 1, right + 1]`.
- Moving `right` when the sum is too small. A too-small sum needs a larger value, so move `left` right.
- Moving `left` when the sum is too large. A too-large sum needs a smaller value, so move `right` left.
- Using `left <= right`. The same element cannot be used twice, so the loop should search only while `left < right`.
- Forgetting that duplicate values are allowed. For example, `[1, 2, 2, 5]` with target `4` should return the two separate `2` positions, `[2, 3]`.
- Adding a hash map unnecessarily. A hash map can solve the original unsorted Two Sum problem, but this sorted variant can be solved with constant auxiliary space.
- Sorting the input again. The input is already sorted, and sorting would change the original positions if the array were not already sorted. This problem's returned positions refer to the given sorted array.

### First-Principles Summary

The brute-force solution asks, "Which pair works?" and then checks every pair.

The two-pointer solution asks a sharper question: "Can this endpoint still be part of any valid pair?"

Sorted order makes that question answerable:

- If the smallest active value plus the largest active value is too small, the smallest active value is hopeless.
- If the smallest active value plus the largest active value is too large, the largest active value is hopeless.

Each step removes one impossible endpoint while keeping the real answer in the remaining interval. That is why the algorithm is both correct and linear.

## Implementation

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`.

## Examples

- `numbers = [2, 7, 11, 15]`, `target = 9` returns `[1, 2]`.
- `numbers = [2, 3, 4]`, `target = 6` returns `[1, 3]`.
- `numbers = [-1, 0]`, `target = -1` returns `[1, 2]`.
- `numbers = [-5, -2, 1, 4, 9]`, `target = 7` returns `[2, 5]`.
- `numbers = [1, 2, 2, 5]`, `target = 4` returns `[2, 3]`.
- See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py` for executable examples and edge cases.
