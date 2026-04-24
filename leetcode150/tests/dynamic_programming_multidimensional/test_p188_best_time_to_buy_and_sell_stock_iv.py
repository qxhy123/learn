from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p188_best_time_to_buy_and_sell_stock_iv import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'k': 2, 'prices': [2, 4, 1]}, 'output': 2}, {'input': {'k': 2, 'prices': [3, 2, 6, 5, 0, 3]}, 'output': 7}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxProfit(**example["input"])
    for example in EXAMPLES:
        result = solution.maxProfit(**example["input"])
        assert result == example["output"]
