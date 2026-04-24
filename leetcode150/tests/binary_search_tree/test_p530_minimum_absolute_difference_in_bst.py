from __future__ import annotations

import pytest

from solutions.binary_search_tree.p530_minimum_absolute_difference_in_bst import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [4, 2, 6, 1, 3]}, 'output': 1}, {'input': {'root': [1, 0, 48, None, None, 12, 49]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().getMinimumDifference(**example["input"])
    for example in EXAMPLES:
        result = solution.getMinimumDifference(**example["input"])
        assert result == example["output"]
