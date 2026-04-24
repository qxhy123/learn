from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p392_is_subsequence.md`."""

    def isSubsequence(self, s: str, t: str) -> bool:
        s_index = 0

        for char in t:
            if s_index == len(s):
                return True
            if s[s_index] == char:
                s_index += 1

        return s_index == len(s)
