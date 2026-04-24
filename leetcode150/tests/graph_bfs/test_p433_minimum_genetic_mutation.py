from __future__ import annotations

import pytest

from solutions.graph_bfs.p433_minimum_genetic_mutation import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'startGene': 'AACCGGTT', 'endGene': 'AACCGGTA', 'bank': ['AACCGGTA']}, 'output': 1}, {'input': {'startGene': 'AACCGGTT', 'endGene': 'AAACGGTA', 'bank': ['AACCGGTA', 'AACCGCTA', 'AAACGGTA']}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minMutation(**example["input"])
    for example in EXAMPLES:
        result = solution.minMutation(**example["input"])
        assert result == example["output"]
