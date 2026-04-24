from __future__ import annotations

import pytest

from solutions.graph_general.p130_surrounded_regions import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]\n[["X"]]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().solve(**example["input"])
    for example in EXAMPLES:
        result = solution.solve(**example["input"])
        assert result == example["output"]
