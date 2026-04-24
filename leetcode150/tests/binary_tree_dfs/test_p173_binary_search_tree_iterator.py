from __future__ import annotations

import pytest

from solutions.binary_tree_dfs.p173_binary_search_tree_iterator import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '["BSTIterator","next","next","hasNext","next","hasNext","next","hasNext","next","hasNext"]\n[[[7,3,15,null,null,9,20]],[],[],[],[],[],[],[],[],[]]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
