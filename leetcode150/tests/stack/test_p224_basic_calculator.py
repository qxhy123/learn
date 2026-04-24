from __future__ import annotations

import pytest

from solutions.stack.p224_basic_calculator import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': '1 + 1'}, 'output': 2}, {'input': {'s': ' 2-1 + 2 '}, 'output': 3}, {'input': {'s': '(1+(4+5+2)-3)+(6+8)'}, 'output': 23}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().calculate(**example["input"])
    for example in EXAMPLES:
        result = solution.calculate(**example["input"])
        assert result == example["output"]
