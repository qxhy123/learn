from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p114_flatten_binary_tree_to_linked_list import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 5, 3, 4, None, 6]}, 'output': [1, None, 2, None, 3, None, 4, None, 5, None, 6]}, {'input': {'root': []}, 'output': []}, {'input': {'root': [0]}, 'output': [0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().flatten(**example["input"])
    for example in EXAMPLES:
        result = solution.flatten(**example["input"])
        assert result == example["output"]
