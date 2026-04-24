from __future__ import annotations

import pytest

from solutions.dynamic_programming_1d.p322_coin_change import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'coins': [1, 2, 5], 'amount': 11}, 'output': 3}, {'input': {'coins': [2], 'amount': 3}, 'output': -1}, {'input': {'coins': [1], 'amount': 0}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().coinChange(**example["input"])
    for example in EXAMPLES:
        result = solution.coinChange(**example["input"])
        assert result == example["output"]
