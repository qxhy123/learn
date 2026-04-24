from __future__ import annotations

import pytest

from solutions.divide_conquer.p023_merge_k_sorted_lists import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'lists': [[1, 4, 5], [1, 3, 4], [2, 6]]}, 'output': [1, 1, 2, 3, 4, 4, 5, 6]}, {'input': {'lists': []}, 'output': []}, {'input': {'lists': [[]]}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().mergeKLists(**example["input"])
    for example in EXAMPLES:
        result = solution.mergeKLists(**example["input"])
        assert result == example["output"]
