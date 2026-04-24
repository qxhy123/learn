from __future__ import annotations

import pytest

from solutions.heap.p215_kth_largest_element_in_an_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [3, 2, 1, 5, 6, 4], 'k': 2}, 'output': 5}, {'input': {'nums': [3, 2, 3, 1, 2, 4, 5, 5, 6], 'k': 4}, 'output': 4}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findKthLargest(**example["input"])
    for example in EXAMPLES:
        result = solution.findKthLargest(**example["input"])
        assert result == example["output"]
