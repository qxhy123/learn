from __future__ import annotations

import pytest

from solutions.bit_manipulation.p137_single_number_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [2, 2, 3, 2]}, 'output': 3}, {'input': {'nums': [0, 1, 0, 1, 0, 1, 99]}, 'output': 99}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().singleNumber(**example["input"])
    for example in EXAMPLES:
        result = solution.singleNumber(**example["input"])
        assert result == example["output"]
