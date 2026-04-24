from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p011_container_with_most_water.md`."""

    def maxArea(self, height: list[int]) -> int:
        left = 0
        right = len(height) - 1
        best_area = 0

        while left < right:
            width = right - left
            current_height = min(height[left], height[right])
            best_area = max(best_area, width * current_height)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return best_area
