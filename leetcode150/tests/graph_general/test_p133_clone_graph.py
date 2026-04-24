from __future__ import annotations

import pytest

from solutions.graph_general.p133_clone_graph import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'adjList': [[2, 4], [1, 3], [2, 4], [1, 3]]}, 'output': [[2, 4], [1, 3], [2, 4], [1, 3]]}, {'input': {'adjList': [[]]}, 'output': [[]]}, {'input': {'adjList': []}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
