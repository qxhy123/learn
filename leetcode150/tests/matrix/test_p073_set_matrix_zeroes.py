from __future__ import annotations

import pytest

from solutions.matrix.p073_set_matrix_zeroes import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'matrix': [[1, 1, 1], [1, 0, 1], [1, 1, 1]]}, 'output': [[1, 0, 1], [0, 0, 0], [1, 0, 1]]}, {'input': {'matrix': [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]}, 'output': [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().setZeroes(**example["input"])
    for example in EXAMPLES:
        result = solution.setZeroes(**example["input"])
        assert result == example["output"]
