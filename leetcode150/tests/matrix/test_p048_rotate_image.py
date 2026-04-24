from __future__ import annotations

import pytest

from solutions.matrix.p048_rotate_image import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, 'output': [[7, 4, 1], [8, 5, 2], [9, 6, 3]]}, {'input': {'matrix': [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]}, 'output': [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rotate(**example["input"])
    for example in EXAMPLES:
        result = solution.rotate(**example["input"])
        assert result == example["output"]
