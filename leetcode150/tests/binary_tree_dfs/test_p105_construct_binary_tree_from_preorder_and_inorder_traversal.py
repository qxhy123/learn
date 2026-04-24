from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p105_construct_binary_tree_from_preorder_and_inorder_traversal import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'preorder': [3, 9, 20, 15, 7], 'inorder': [9, 3, 15, 20, 7]}, 'output': [3, 9, 20, None, None, 15, 7]}, {'input': {'preorder': [-1], 'inorder': [-1]}, 'output': [-1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().buildTree(**example["input"])
    for example in EXAMPLES:
        result = solution.buildTree(**example["input"])
        assert result == example["output"]
