from __future__ import annotations

import pytest

from solutions.binary_search_tree.p098_validate_binary_search_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [2, 1, 3]}, 'output': True}, {'input': {'root': [5, 1, 4, None, None, 3, 6]}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isValidBST(**example["input"])
    for example in EXAMPLES:
        result = solution.isValidBST(**example["input"])
        assert result == example["output"]
