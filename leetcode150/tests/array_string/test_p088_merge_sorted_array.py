from __future__ import annotations

import pytest

from solutions.array_string.p088_merge_sorted_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums1': [1, 2, 3, 0, 0, 0], 'm': 3, 'nums2': [2, 5, 6], 'n': 3}, 'output': [1, 2, 2, 3, 5, 6]}, {'input': {'nums1': [1], 'm': 1, 'nums2': [], 'n': 0}, 'output': [1]}, {'input': {'nums1': [0], 'm': 0, 'nums2': [1], 'n': 1}, 'output': [1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().merge(**example["input"])
    for example in EXAMPLES:
        result = solution.merge(**example["input"])
        assert result == example["output"]
