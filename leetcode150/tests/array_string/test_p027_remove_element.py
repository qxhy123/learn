from __future__ import annotations

import pytest

from solutions.array_string.p027_remove_element import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [3, 2, 2, 3], 'val': 3}, 'output': '2, nums = [2,2,_,_]'}, {'input': {'nums': [0, 1, 2, 2, 3, 0, 4, 2], 'val': 2}, 'output': '5, nums = [0,1,4,0,3,_,_,_]'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().removeElement(**example["input"])
    for example in EXAMPLES:
        result = solution.removeElement(**example["input"])
        assert result == example["output"]
