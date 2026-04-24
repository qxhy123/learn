from __future__ import annotations

import pytest

from solutions.two_pointers.p125_valid_palindrome import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'A man, a plan, a canal: Panama'}, 'output': True}, {'input': {'s': 'race a car'}, 'output': False}, {'input': {'s': ' '}, 'output': True}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isPalindrome(**example["input"])
    for example in EXAMPLES:
        result = solution.isPalindrome(**example["input"])
        assert result == example["output"]
