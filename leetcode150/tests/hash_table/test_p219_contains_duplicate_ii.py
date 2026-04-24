from __future__ import annotations

import pytest

from solutions.hash_table.p219_contains_duplicate_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3, 1], 'k': 3}, 'output': True}, {'input': {'nums': [1, 0, 1, 1], 'k': 1}, 'output': True}, {'input': {'nums': [1, 2, 3, 1, 2, 3], 'k': 2}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().containsNearbyDuplicate(**example["input"])
    for example in EXAMPLES:
        result = solution.containsNearbyDuplicate(**example["input"])
        assert result == example["output"]
