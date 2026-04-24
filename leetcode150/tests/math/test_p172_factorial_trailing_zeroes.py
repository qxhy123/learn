from __future__ import annotations

import pytest

from solutions.math.p172_factorial_trailing_zeroes import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 3}, 'output': 0}, {'input': {'n': 5}, 'output': 1}, {'input': {'n': 0}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().trailingZeroes(**example["input"])
    for example in EXAMPLES:
        result = solution.trailingZeroes(**example["input"])
        assert result == example["output"]
