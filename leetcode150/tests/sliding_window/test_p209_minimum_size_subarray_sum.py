from __future__ import annotations

import pytest

from solutions.sliding_window.p209_minimum_size_subarray_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'target': 7, 'nums': [2, 3, 1, 2, 4, 3]}, 'output': 2}, {'input': {'target': 4, 'nums': [1, 4, 4]}, 'output': 1}, {'input': {'target': 11, 'nums': [1, 1, 1, 1, 1, 1, 1, 1]}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minSubArrayLen(**example["input"])
    for example in EXAMPLES:
        result = solution.minSubArrayLen(**example["input"])
        assert result == example["output"]
