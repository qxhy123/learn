from __future__ import annotations

import pytest

from solutions.trie.p208_implement_trie_prefix_tree import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '["Trie","insert","search","search","startsWith","insert","search"]\n[[],["apple"],["apple"],["app"],["app"],["app"],["app"]]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
