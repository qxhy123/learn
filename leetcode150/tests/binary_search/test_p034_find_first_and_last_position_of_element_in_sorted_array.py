from __future__ import annotations

import pytest

from solutions.binary_search.p034_find_first_and_last_position_of_element_in_sorted_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [5, 7, 7, 8, 8, 10], 'target': 8}, 'output': [3, 4]}, {'input': {'nums': [5, 7, 7, 8, 8, 10], 'target': 6}, 'output': [-1, -1]}, {'input': {'nums': [], 'target': 0}, 'output': [-1, -1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().searchRange(**example["input"])
    for example in EXAMPLES:
        result = solution.searchRange(**example["input"])
        assert result == example["output"]
