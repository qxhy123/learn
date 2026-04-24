from __future__ import annotations

import pytest

from solutions.array_string.p028_find_the_index_of_the_first_occurrence_in_a_string import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'haystack': 'sadbutsad', 'needle': 'sad'}, 'output': 0}, {'input': {'haystack': 'leetcode', 'needle': 'leeto'}, 'output': -1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().strStr(**example["input"])
    for example in EXAMPLES:
        result = solution.strStr(**example["input"])
        assert result == example["output"]
