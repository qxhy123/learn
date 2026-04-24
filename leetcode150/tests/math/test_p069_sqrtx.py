from __future__ import annotations

import pytest

from solutions.math.p069_sqrtx import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'x': 4}, 'output': 2}, {'input': {'x': 8}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().mySqrt(**example["input"])
    for example in EXAMPLES:
        result = solution.mySqrt(**example["input"])
        assert result == example["output"]
