from __future__ import annotations

import pytest

from solutions.binary_search.p074_search_a_2d_matrix import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'matrix': [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 'target': 3}, 'output': True}, {'input': {'matrix': [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 'target': 13}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().searchMatrix(**example["input"])
    for example in EXAMPLES:
        result = solution.searchMatrix(**example["input"])
        assert result == example["output"]
