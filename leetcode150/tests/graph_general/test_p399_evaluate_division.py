from __future__ import annotations

import pytest

from solutions.graph_general.p399_evaluate_division import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'equations': [['a', 'b'], ['b', 'c']], 'values': [2.0, 3.0], 'queries': [['a', 'c'], ['b', 'a'], ['a', 'e'], ['a', 'a'], ['x', 'x']]}, 'output': [6.0, 0.5, -1.0, 1.0, -1.0]}, {'input': {'equations': [['a', 'b'], ['b', 'c'], ['bc', 'cd']], 'values': [1.5, 2.5, 5.0], 'queries': [['a', 'c'], ['c', 'b'], ['bc', 'cd'], ['cd', 'bc']]}, 'output': [3.75, 0.4, 5.0, 0.2]}, {'input': {'equations': [['a', 'b']], 'values': [0.5], 'queries': [['a', 'b'], ['b', 'a'], ['a', 'c'], ['x', 'y']]}, 'output': [0.5, 2.0, -1.0, -1.0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().calcEquation(**example["input"])
    for example in EXAMPLES:
        result = solution.calcEquation(**example["input"])
        assert result == example["output"]
