from __future__ import annotations

import pytest

from solutions.kadane.p053_maximum_subarray import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, 'output': 6}, {'input': {'nums': [1]}, 'output': 1}, {'input': {'nums': [5, 4, -1, 7, 8]}, 'output': 23}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxSubArray(**example["input"])
    for example in EXAMPLES:
        result = solution.maxSubArray(**example["input"])
        assert result == example["output"]
