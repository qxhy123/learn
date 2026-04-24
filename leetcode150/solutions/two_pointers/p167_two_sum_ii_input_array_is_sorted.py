from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md`."""

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
