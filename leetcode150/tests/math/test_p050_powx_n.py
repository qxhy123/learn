from __future__ import annotations

import pytest

from solutions.math.p050_powx_n import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'x': 2.0, 'n': 10}, 'output': 1024.0}, {'input': {'x': 2.1, 'n': 3}, 'output': 9.261}, {'input': {'x': 2.0, 'n': -2}, 'output': 0.25}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().myPow(**example["input"])
    for example in EXAMPLES:
        result = solution.myPow(**example["input"])
        assert result == example["output"]
