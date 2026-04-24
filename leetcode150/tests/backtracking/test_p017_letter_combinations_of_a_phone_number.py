from __future__ import annotations

import pytest

from solutions.backtracking.p017_letter_combinations_of_a_phone_number import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'digits': '23'}, 'output': ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']}, {'input': {'digits': '2'}, 'output': ['a', 'b', 'c']}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().letterCombinations(**example["input"])
    for example in EXAMPLES:
        result = solution.letterCombinations(**example["input"])
        assert result == example["output"]
