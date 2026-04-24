from __future__ import annotations

import pytest

from solutions.array_string.p134_gas_station import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'gas': [1, 2, 3, 4, 5], 'cost': [3, 4, 5, 1, 2]}, 'output': 3}, {'input': {'gas': [2, 3, 4], 'cost': [3, 4, 3]}, 'output': -1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().canCompleteCircuit(**example["input"])
    for example in EXAMPLES:
        result = solution.canCompleteCircuit(**example["input"])
        assert result == example["output"]
