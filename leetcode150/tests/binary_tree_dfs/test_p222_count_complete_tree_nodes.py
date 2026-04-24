from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p222_count_complete_tree_nodes import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 3, 4, 5, 6]}, 'output': 6}, {'input': {'root': []}, 'output': 0}, {'input': {'root': [1]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().countNodes(**example["input"])
    for example in EXAMPLES:
        result = solution.countNodes(**example["input"])
        assert result == example["output"]
