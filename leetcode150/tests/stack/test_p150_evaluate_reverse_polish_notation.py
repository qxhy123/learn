from __future__ import annotations

import pytest

from solutions.stack.p150_evaluate_reverse_polish_notation import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'tokens': ['2', '1', '+', '3', '*']}, 'output': 9}, {'input': {'tokens': ['4', '13', '5', '/', '+']}, 'output': 6}, {'input': {'tokens': ['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']}, 'output': 22}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().evalRPN(**example["input"])
    for example in EXAMPLES:
        result = solution.evalRPN(**example["input"])
        assert result == example["output"]
