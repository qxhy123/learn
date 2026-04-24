from __future__ import annotations

import pytest

from solutions.array_string.p238_product_of_array_except_self import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3, 4]}, 'output': [24, 12, 8, 6]}, {'input': {'nums': [-1, 1, 0, -3, 3]}, 'output': [0, 0, 9, 0, 0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().productExceptSelf(**example["input"])
    for example in EXAMPLES:
        result = solution.productExceptSelf(**example["input"])
        assert result == example["output"]
