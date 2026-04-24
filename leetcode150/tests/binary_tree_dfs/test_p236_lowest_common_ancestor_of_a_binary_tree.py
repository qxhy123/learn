from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p236_lowest_common_ancestor_of_a_binary_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 'p': 5, 'q': 1}, 'output': 3}, {'input': {'root': [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 'p': 5, 'q': 4}, 'output': 5}, {'input': {'root': [1, 2], 'p': 1, 'q': 2}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().lowestCommonAncestor(**example["input"])
    for example in EXAMPLES:
        result = solution.lowestCommonAncestor(**example["input"])
        assert result == example["output"]
