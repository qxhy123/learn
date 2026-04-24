from __future__ import annotations

import pytest

from solutions.two_pointers.p015_3sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [-1, 0, 1, 2, -1, -4]}, 'output': [[-1, -1, 2], [-1, 0, 1]]}, {'input': {'nums': [0, 1, 1]}, 'output': []}, {'input': {'nums': [0, 0, 0]}, 'output': [[0, 0, 0]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().threeSum(**example["input"])
    for example in EXAMPLES:
        result = solution.threeSum(**example["input"])
        assert result == example["output"]
