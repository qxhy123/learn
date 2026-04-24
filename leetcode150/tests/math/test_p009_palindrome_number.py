from __future__ import annotations

import pytest

from solutions.math.p009_palindrome_number import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'x': 121}, 'output': True}, {'input': {'x': -121}, 'output': False}, {'input': {'x': 10}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isPalindrome(**example["input"])
    for example in EXAMPLES:
        result = solution.isPalindrome(**example["input"])
        assert result == example["output"]
