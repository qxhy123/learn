from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p226_invert_binary_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [4, 2, 7, 1, 3, 6, 9]}, 'output': [4, 7, 2, 9, 6, 3, 1]}, {'input': {'root': [2, 1, 3]}, 'output': [2, 3, 1]}, {'input': {'root': []}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().invertTree(**example["input"])
    for example in EXAMPLES:
        result = solution.invertTree(**example["input"])
        assert result == example["output"]
