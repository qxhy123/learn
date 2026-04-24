from __future__ import annotations

import pytest

from solutions.binary_tree_bfs.p103_binary_tree_zigzag_level_order_traversal import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [3, 9, 20, None, None, 15, 7]}, 'output': [[3], [20, 9], [15, 7]]}, {'input': {'root': [1]}, 'output': [[1]]}, {'input': {'root': []}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().zigzagLevelOrder(**example["input"])
    for example in EXAMPLES:
        result = solution.zigzagLevelOrder(**example["input"])
        assert result == example["output"]
