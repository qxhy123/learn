from __future__ import annotations

import pytest

from solutions.stack.p020_valid_parentheses import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"()"\n"()[]{}"\n"(]"\n"([])"\n"([)]"'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isValid(**example["input"])
    for example in EXAMPLES:
        result = solution.isValid(**example["input"])
        assert result == example["output"]
