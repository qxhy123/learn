from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p124_binary_tree_maximum_path_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 3]}, 'output': 6}, {'input': {'root': [-10, 9, 20, None, None, 15, 7]}, 'output': 42}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxPathSum(**example["input"])
    for example in EXAMPLES:
        result = solution.maxPathSum(**example["input"])
        assert result == example["output"]
