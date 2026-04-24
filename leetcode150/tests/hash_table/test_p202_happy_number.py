from __future__ import annotations

import pytest

from solutions.hash_table.p202_happy_number import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 19}, 'output': True}, {'input': {'n': 2}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isHappy(**example["input"])
    for example in EXAMPLES:
        result = solution.isHappy(**example["input"])
        assert result == example["output"]
