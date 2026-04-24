from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p015_3sum.md`."""

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        triplets: list[list[int]] = []

        for fixed_index in range(len(nums) - 2):
            fixed_value = nums[fixed_index]
            if fixed_index > 0 and fixed_value == nums[fixed_index - 1]:
                continue
            if fixed_value > 0:
                break

            left = fixed_index + 1
            right = len(nums) - 1

            while left < right:
                current_sum = fixed_value + nums[left] + nums[right]
                if current_sum == 0:
                    triplets.append([fixed_value, nums[left], nums[right]])
                    left += 1
                    right -= 1

                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif current_sum < 0:
                    left += 1
                else:
                    right -= 1

        return triplets
