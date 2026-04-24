from __future__ import annotations

import pytest

from solutions.linked_list.p019_remove_nth_node_from_end_of_list import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 2, 3, 4, 5], 'n': 2}, 'output': [1, 2, 3, 5]}, {'input': {'head': [1], 'n': 1}, 'output': []}, {'input': {'head': [1, 2], 'n': 1}, 'output': [1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().removeNthFromEnd(**example["input"])
    for example in EXAMPLES:
        result = solution.removeNthFromEnd(**example["input"])
        assert result == example["output"]
