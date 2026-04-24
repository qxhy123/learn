from __future__ import annotations

import pytest

from solutions.array_string.p013_roman_to_integer import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'III'}, 'output': 3}, {'input': {'s': 'LVIII'}, 'output': 58}, {'input': {'s': 'MCMXCIV'}, 'output': 1994}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().romanToInt(**example["input"])
    for example in EXAMPLES:
        result = solution.romanToInt(**example["input"])
        assert result == example["output"]
