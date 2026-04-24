from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p112_path_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1], 'targetSum': 22}, 'output': True}, {'input': {'root': [1, 2, 3], 'targetSum': 5}, 'output': False}, {'input': {'root': [], 'targetSum': 0}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().hasPathSum(**example["input"])
    for example in EXAMPLES:
        result = solution.hasPathSum(**example["input"])
        assert result == example["output"]
