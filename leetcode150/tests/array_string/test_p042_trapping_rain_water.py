from __future__ import annotations

import pytest

from solutions.array_string.p042_trapping_rain_water import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'height': [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]}, 'output': 6}, {'input': {'height': [4, 2, 0, 3, 2, 5]}, 'output': 9}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().trap(**example["input"])
    for example in EXAMPLES:
        result = solution.trap(**example["input"])
        assert result == example["output"]
