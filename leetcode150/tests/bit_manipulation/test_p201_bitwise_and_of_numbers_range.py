from __future__ import annotations

import pytest

from solutions.bit_manipulation.p201_bitwise_and_of_numbers_range import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'left': 5, 'right': 7}, 'output': 4}, {'input': {'left': 0, 'right': 0}, 'output': 0}, {'input': {'left': 1, 'right': 2147483647}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rangeBitwiseAnd(**example["input"])
    for example in EXAMPLES:
        result = solution.rangeBitwiseAnd(**example["input"])
        assert result == example["output"]
