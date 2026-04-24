from __future__ import annotations

import pytest

from solutions.kadane.p918_maximum_sum_circular_subarray import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, -2, 3, -2]}, 'output': 3}, {'input': {'nums': [5, -3, 5]}, 'output': 10}, {'input': {'nums': [-3, -2, -3]}, 'output': -2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxSubarraySumCircular(**example["input"])
    for example in EXAMPLES:
        result = solution.maxSubarraySumCircular(**example["input"])
        assert result == example["output"]
