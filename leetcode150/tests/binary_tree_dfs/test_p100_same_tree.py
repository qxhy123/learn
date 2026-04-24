from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p100_same_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'p': [1, 2, 3], 'q': [1, 2, 3]}, 'output': True}, {'input': {'p': [1, 2], 'q': [1, None, 2]}, 'output': False}, {'input': {'p': [1, 2, 1], 'q': [1, 1, 2]}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isSameTree(**example["input"])
    for example in EXAMPLES:
        result = solution.isSameTree(**example["input"])
        assert result == example["output"]
