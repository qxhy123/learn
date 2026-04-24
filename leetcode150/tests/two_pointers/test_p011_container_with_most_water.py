from __future__ import annotations

import pytest

from solutions.two_pointers.p011_container_with_most_water import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'height': [1, 8, 6, 2, 5, 4, 8, 3, 7]}, 'output': 49}, {'input': {'height': [1, 1]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maxArea(**example["input"])
    for example in EXAMPLES:
        result = solution.maxArea(**example["input"])
        assert result == example["output"]
