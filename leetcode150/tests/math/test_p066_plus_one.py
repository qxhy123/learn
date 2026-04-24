from __future__ import annotations

import pytest

from solutions.math.p066_plus_one import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'digits': [1, 2, 3]}, 'output': [1, 2, 4]}, {'input': {'digits': [4, 3, 2, 1]}, 'output': [4, 3, 2, 2]}, {'input': {'digits': [9]}, 'output': [1, 0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().plusOne(**example["input"])
    for example in EXAMPLES:
        result = solution.plusOne(**example["input"])
        assert result == example["output"]
