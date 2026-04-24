from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p101_symmetric_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 2, 3, 4, 4, 3]}, 'output': True}, {'input': {'root': [1, 2, 2, None, 3, None, 3]}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isSymmetric(**example["input"])
    for example in EXAMPLES:
        result = solution.isSymmetric(**example["input"])
        assert result == example["output"]
