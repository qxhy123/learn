from __future__ import annotations

import pytest

from solutions.array_string.p135_candy import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'ratings': [1, 0, 2]}, 'output': 5}, {'input': {'ratings': [1, 2, 2]}, 'output': 4}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().candy(**example["input"])
    for example in EXAMPLES:
        result = solution.candy(**example["input"])
        assert result == example["output"]
