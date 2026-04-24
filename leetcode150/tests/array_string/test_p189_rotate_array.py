from __future__ import annotations

import pytest

from solutions.array_string.p189_rotate_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3, 4, 5, 6, 7], 'k': 3}, 'output': [5, 6, 7, 1, 2, 3, 4]}, {'input': {'nums': [-1, -100, 3, 99], 'k': 2}, 'output': [3, 99, -1, -100]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rotate(**example["input"])
    for example in EXAMPLES:
        result = solution.rotate(**example["input"])
        assert result == example["output"]
