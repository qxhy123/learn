from __future__ import annotations

import pytest

from solutions.matrix.p054_spiral_matrix import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, 'output': [1, 2, 3, 6, 9, 8, 7, 4, 5]}, {'input': {'matrix': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]}, 'output': [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().spiralOrder(**example["input"])
    for example in EXAMPLES:
        result = solution.spiralOrder(**example["input"])
        assert result == example["output"]
