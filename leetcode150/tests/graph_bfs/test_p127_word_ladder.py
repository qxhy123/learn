from __future__ import annotations

import pytest

from solutions.graph_bfs.p127_word_ladder import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'beginWord': 'hit', 'endWord': 'cog', 'wordList': ['hot', 'dot', 'dog', 'lot', 'log', 'cog']}, 'output': 5}, {'input': {'beginWord': 'hit', 'endWord': 'cog', 'wordList': ['hot', 'dot', 'dog', 'lot', 'log']}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().ladderLength(**example["input"])
    for example in EXAMPLES:
        result = solution.ladderLength(**example["input"])
        assert result == example["output"]
