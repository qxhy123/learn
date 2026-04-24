from __future__ import annotations

import pytest

from solutions.binary_search.p033_search_in_rotated_sorted_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [4, 5, 6, 7, 0, 1, 2], 'target': 0}, 'output': 4}, {'input': {'nums': [4, 5, 6, 7, 0, 1, 2], 'target': 3}, 'output': -1}, {'input': {'nums': [1], 'target': 0}, 'output': -1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().search(**example["input"])
    for example in EXAMPLES:
        result = solution.search(**example["input"])
        assert result == example["output"]
