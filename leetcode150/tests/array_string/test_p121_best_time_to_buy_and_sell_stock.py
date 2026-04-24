from __future__ import annotations

import pytest

from solutions.array_string.p121_best_time_to_buy_and_sell_stock import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'prices': [7, 1, 5, 3, 6, 4]}, 'output': 5}, {'input': {'prices': [7, 6, 4, 3, 1]}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxProfit(**example["input"])
    for example in EXAMPLES:
        result = solution.maxProfit(**example["input"])
        assert result == example["output"]
