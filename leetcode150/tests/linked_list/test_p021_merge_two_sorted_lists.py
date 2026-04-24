from __future__ import annotations

import pytest

from solutions.linked_list.p021_merge_two_sorted_lists import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'list1': [1, 2, 4], 'list2': [1, 3, 4]}, 'output': [1, 1, 2, 3, 4, 4]}, {'input': {'list1': [], 'list2': []}, 'output': []}, {'input': {'list1': [], 'list2': [0]}, 'output': [0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().mergeTwoLists(**example["input"])
    for example in EXAMPLES:
        result = solution.mergeTwoLists(**example["input"])
        assert result == example["output"]
