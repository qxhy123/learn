from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p129_sum_root_to_leaf_numbers import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 3]}, 'output': 25}, {'input': {'root': [4, 9, 0, 5, 1]}, 'output': 1026}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().sumNumbers(**example["input"])
    for example in EXAMPLES:
        result = solution.sumNumbers(**example["input"])
        assert result == example["output"]
