from __future__ import annotations

import pytest

from solutions.binary_tree_bfs.p637_average_of_levels_in_binary_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'root': [3, 9, 20, None, None, 15, 7]}, 'output': [3.0, 14.5, 11.0]}, {'input': {'root': [3, 9, 20, 15, 7]}, 'output': [3.0, 14.5, 11.0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().averageOfLevels(**example["input"])
    for example in EXAMPLES:
        result = solution.averageOfLevels(**example["input"])
        assert result == example["output"]
