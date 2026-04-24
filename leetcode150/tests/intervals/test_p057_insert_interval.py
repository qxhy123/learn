from __future__ import annotations

import pytest

from solutions.intervals.p057_insert_interval import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'intervals': [[1, 3], [6, 9]], 'newInterval': [2, 5]}, 'output': [[1, 5], [6, 9]]}, {'input': {'intervals': [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], 'newInterval': [4, 8]}, 'output': [[1, 2], [3, 10], [12, 16]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().insert(**example["input"])
    for example in EXAMPLES:
        result = solution.insert(**example["input"])
        assert result == example["output"]
