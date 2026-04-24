from __future__ import annotations

import pytest

from solutions.math.p149_max_points_on_a_line import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'points': [[1, 1], [2, 2], [3, 3]]}, 'output': 3}, {'input': {'points': [[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]}, 'output': 4}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxPoints(**example["input"])
    for example in EXAMPLES:
        result = solution.maxPoints(**example["input"])
        assert result == example["output"]
