from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p005_longest_palindromic_substring import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'babad'}, 'output': 'bab'}, {'input': {'s': 'cbbd'}, 'output': 'bb'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().longestPalindrome(**example["input"])
    for example in EXAMPLES:
        result = solution.longestPalindrome(**example["input"])
        assert result == example["output"]
