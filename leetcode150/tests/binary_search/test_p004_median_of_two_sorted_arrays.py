from __future__ import annotations

import pytest

from solutions.binary_search.p004_median_of_two_sorted_arrays import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums1': [1, 3], 'nums2': [2]}, 'output': 2.0}, {'input': {'nums1': [1, 2], 'nums2': [3, 4]}, 'output': 2.5}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findMedianSortedArrays(**example["input"])
    for example in EXAMPLES:
        result = solution.findMedianSortedArrays(**example["input"])
        assert result == example["output"]
