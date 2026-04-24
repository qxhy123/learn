from __future__ import annotations

import pytest

from solutions.backtracking.p022_generate_parentheses import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 3}, 'output': ['((()))', '(()())', '(())()', '()(())', '()()()']}, {'input': {'n': 1}, 'output': ['()']}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().generateParenthesis(**example["input"])
    for example in EXAMPLES:
        result = solution.generateParenthesis(**example["input"])
        assert result == example["output"]
