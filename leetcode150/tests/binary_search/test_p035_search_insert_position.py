from __future__ import annotations

import pytest

from solutions.binary_search.p035_search_insert_position import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 3, 5, 6], 'target': 5}, 'output': 2}, {'input': {'nums': [1, 3, 5, 6], 'target': 2}, 'output': 1}, {'input': {'nums': [1, 3, 5, 6], 'target': 7}, 'output': 4}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().searchInsert(**example["input"])
    for example in EXAMPLES:
        result = solution.searchInsert(**example["input"])
        assert result == example["output"]
