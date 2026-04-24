from __future__ import annotations

import pytest

from solutions.array_string.p122_best_time_to_buy_and_sell_stock_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'prices': [7, 1, 5, 3, 6, 4]}, 'output': 7}, {'input': {'prices': [1, 2, 3, 4, 5]}, 'output': 4}, {'input': {'prices': [7, 6, 4, 3, 1]}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxProfit(**example["input"])
    for example in EXAMPLES:
        result = solution.maxProfit(**example["input"])
        assert result == example["output"]
