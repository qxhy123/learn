from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p117_populating_next_right_pointers_in_each_node_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [1, 2, 3, 4, 5, None, 7]}, 'output': '[1,#,2,3,#,4,5,7,#]'}, {'input': {'root': []}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
