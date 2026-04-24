from __future__ import annotations

import pytest

from solutions.intervals.p452_minimum_number_of_arrows_to_burst_balloons import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'points': [[10, 16], [2, 8], [1, 6], [7, 12]]}, 'output': 2}, {'input': {'points': [[1, 2], [3, 4], [5, 6], [7, 8]]}, 'output': 4}, {'input': {'points': [[1, 2], [2, 3], [3, 4], [4, 5]]}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findMinArrowShots(**example["input"])
    for example in EXAMPLES:
        result = solution.findMinArrowShots(**example["input"])
        assert result == example["output"]
