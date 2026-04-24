from __future__ import annotations

import pytest

from solutions.binary_tree_bfs.p199_binary_tree_right_side_view import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '[1,2,3,null,5,null,4]\n[1,2,3,4,null,null,null,5]\n[1,null,3]\n[]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rightSideView(**example["input"])
    for example in EXAMPLES:
        result = solution.rightSideView(**example["input"])
        assert result == example["output"]
