from __future__ import annotations

import pytest

from solutions.binary_search_tree.p230_kth_smallest_element_in_a_bst import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [3, 1, 4, None, 2], 'k': 1}, 'output': 1}, {'input': {'root': [5, 3, 6, 2, 4, None, None, 1], 'k': 3}, 'output': 3}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().kthSmallest(**example["input"])
    for example in EXAMPLES:
        result = solution.kthSmallest(**example["input"])
        assert result == example["output"]
