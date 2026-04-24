from __future__ import annotations

import pytest

from solutions.divide_conquer.p108_convert_sorted_array_to_binary_search_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [-10, -3, 0, 5, 9]}, 'output': [0, -3, 9, -10, None, 5]}, {'input': {'nums': [1, 3]}, 'output': [3, 1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().sortedArrayToBST(**example["input"])
    for example in EXAMPLES:
        result = solution.sortedArrayToBST(**example["input"])
        assert result == example["output"]
