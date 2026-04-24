from __future__ import annotations

import pytest

from solutions.sliding_window.p030_substring_with_concatenation_of_all_words import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"barfoothefoobarman"\n["foo","bar"]\n"wordgoodgoodgoodbestword"\n["word","good","best","word"]\n"barfoofoobarthefoobarman"\n["bar","foo","the"]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findSubstring(**example["input"])
    for example in EXAMPLES:
        result = solution.findSubstring(**example["input"])
        assert result == example["output"]
