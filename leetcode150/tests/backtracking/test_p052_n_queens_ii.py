from __future__ import annotations

import pytest

from solutions.backtracking.p052_n_queens_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 4}, 'output': 2}, {'input': {'n': 1}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().totalNQueens(**example["input"])
    for example in EXAMPLES:
        result = solution.totalNQueens(**example["input"])
        assert result == example["output"]
