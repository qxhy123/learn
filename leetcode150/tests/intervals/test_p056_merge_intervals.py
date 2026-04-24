from __future__ import annotations

import pytest

from solutions.intervals.p056_merge_intervals import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'intervals': [[1, 3], [2, 6], [8, 10], [15, 18]]}, 'output': [[1, 6], [8, 10], [15, 18]]}, {'input': {'intervals': [[1, 4], [4, 5]]}, 'output': [[1, 5]]}, {'input': {'intervals': [[4, 7], [1, 4]]}, 'output': [[1, 7]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().merge(**example["input"])
    for example in EXAMPLES:
        result = solution.merge(**example["input"])
        assert result == example["output"]
