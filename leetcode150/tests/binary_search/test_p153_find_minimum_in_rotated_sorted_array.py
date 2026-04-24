from __future__ import annotations

import pytest

from solutions.binary_search.p153_find_minimum_in_rotated_sorted_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [3, 4, 5, 1, 2]}, 'output': 1}, {'input': {'nums': [4, 5, 6, 7, 0, 1, 2]}, 'output': 0}, {'input': {'nums': [11, 13, 15, 17]}, 'output': 11}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findMin(**example["input"])
    for example in EXAMPLES:
        result = solution.findMin(**example["input"])
        assert result == example["output"]
