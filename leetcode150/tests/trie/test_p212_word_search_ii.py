from __future__ import annotations

import pytest

from solutions.trie.p212_word_search_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'board': [['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], 'words': ['oath', 'pea', 'eat', 'rain']}, 'output': ['eat', 'oath']}, {'input': {'board': [['a', 'b'], ['c', 'd']], 'words': ['abcb']}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findWords(**example["input"])
    for example in EXAMPLES:
        result = solution.findWords(**example["input"])
        assert result == example["output"]
