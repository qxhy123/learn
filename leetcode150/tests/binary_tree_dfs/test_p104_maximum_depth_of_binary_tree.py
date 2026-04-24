from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p104_maximum_depth_of_binary_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [3, 9, 20, None, None, 15, 7]}, 'output': 3}, {'input': {'root': [1, None, 2]}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxDepth(**example["input"])
    for example in EXAMPLES:
        result = solution.maxDepth(**example["input"])
        assert result == example["output"]
