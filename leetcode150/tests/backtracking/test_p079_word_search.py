from __future__ import annotations

import pytest

from solutions.backtracking.p079_word_search import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'ABCCED'}, 'output': True}, {'input': {'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'SEE'}, 'output': True}, {'input': {'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'ABCB'}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().exist(**example["input"])
    for example in EXAMPLES:
        result = solution.exist(**example["input"])
        assert result == example["output"]
