from __future__ import annotations

import pytest

from solutions.hash_table.p001_two_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [2, 7, 11, 15], 'target': 9}, 'output': [0, 1]}, {'input': {'nums': [3, 2, 4], 'target': 6}, 'output': [1, 2]}, {'input': {'nums': [3, 3], 'target': 6}, 'output': [0, 1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().twoSum(**example["input"])
    for example in EXAMPLES:
        result = solution.twoSum(**example["input"])
        assert result == example["output"]
