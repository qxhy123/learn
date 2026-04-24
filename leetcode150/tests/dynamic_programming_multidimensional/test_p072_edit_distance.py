from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p072_edit_distance import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'word1': 'horse', 'word2': 'ros'}, 'output': 3}, {'input': {'word1': 'intention', 'word2': 'execution'}, 'output': 5}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minDistance(**example["input"])
    for example in EXAMPLES:
        result = solution.minDistance(**example["input"])
        assert result == example["output"]
