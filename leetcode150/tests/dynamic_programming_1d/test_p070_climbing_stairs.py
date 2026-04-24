from __future__ import annotations

import pytest

from solutions.dynamic_programming_1d.p070_climbing_stairs import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 2}, 'output': 2}, {'input': {'n': 3}, 'output': 3}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().climbStairs(**example["input"])
    for example in EXAMPLES:
        result = solution.climbStairs(**example["input"])
        assert result == example["output"]
