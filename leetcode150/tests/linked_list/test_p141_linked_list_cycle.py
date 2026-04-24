from __future__ import annotations

import pytest

from solutions.linked_list.p141_linked_list_cycle import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [3, 2, 0, -4], 'pos': 1}, 'output': True}, {'input': {'head': [1, 2], 'pos': 0}, 'output': True}, {'input': {'head': [1], 'pos': -1}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().hasCycle(**example["input"])
    for example in EXAMPLES:
        result = solution.hasCycle(**example["input"])
        assert result == example["output"]
