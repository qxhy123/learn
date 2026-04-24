from __future__ import annotations

import pytest

from solutions.graph_general.p200_number_of_islands import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'grid': [['1', '1', '1', '1', '0'], ['1', '1', '0', '1', '0'], ['1', '1', '0', '0', '0'], ['0', '0', '0', '0', '0']]}, 'output': 1}, {'input': {'grid': [['1', '1', '0', '0', '0'], ['1', '1', '0', '0', '0'], ['0', '0', '1', '0', '0'], ['0', '0', '0', '1', '1']]}, 'output': 3}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().numIslands(**example["input"])
    for example in EXAMPLES:
        result = solution.numIslands(**example["input"])
        assert result == example["output"]
