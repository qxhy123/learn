from __future__ import annotations

import pytest

from solutions.linked_list.p025_reverse_nodes_in_k_group import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 2, 3, 4, 5], 'k': 2}, 'output': [2, 1, 4, 3, 5]}, {'input': {'head': [1, 2, 3, 4, 5], 'k': 3}, 'output': [3, 2, 1, 4, 5]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().reverseKGroup(**example["input"])
    for example in EXAMPLES:
        result = solution.reverseKGroup(**example["input"])
        assert result == example["output"]
