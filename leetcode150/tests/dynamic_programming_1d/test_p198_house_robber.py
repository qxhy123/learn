from __future__ import annotations

import pytest

from solutions.dynamic_programming_1d.p198_house_robber import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3, 1]}, 'output': 4}, {'input': {'nums': [2, 7, 9, 3, 1]}, 'output': 12}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rob(**example["input"])
    for example in EXAMPLES:
        result = solution.rob(**example["input"])
        assert result == example["output"]
