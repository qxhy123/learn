from __future__ import annotations

"""125. Valid Palindrome.

Goal
----
Decide whether the string reads the same from both ends after applying the
problem's normalization rules:

1. keep only alphanumeric characters;
2. compare letters case-insensitively.

The direct beginner solution is:

    cleaned = [ch.lower() for ch in s if ch.isalnum()]
    return cleaned == cleaned[::-1]

That is perfectly valid logically, but it builds an extra list. This file uses
the interview-oriented two-pointer version: perform the same filtering lazily
while scanning from both ends of the original string.

Key idea
--------
A palindrome is a sequence of mirrored pairs. The first meaningful character
must match the last meaningful character, the second meaningful character must
match the second-to-last, and so on.

`left` always searches for the next meaningful character from the front.
`right` always searches for the next meaningful character from the back.

After both pointers land on meaningful characters, they represent the next pair
that must match in the normalized string.
"""


class Solution:
    """LeetCode-style solution container."""

    def isPalindrome(self, s: str) -> bool:
        """Return True if `s` is a palindrome after normalization.

        Walk-through on "A man, a plan, a canal: Panama":
        - `left` starts at 'A', `right` starts at 'a'; compare 'a' == 'a'.
        - punctuation and spaces are skipped whenever a pointer reaches them.
        - every meaningful mirrored pair matches, so the pointers eventually
          cross and the method returns True.

        Walk-through on "0P":
        - both characters are meaningful;
        - compare '0' with 'p'; they differ, so return False immediately.

        Invariant:
            At the start of each outer loop iteration, all meaningful pairs
            outside the current [left, right] window have already matched.

        Complexity:
            Time: O(n), because each pointer only moves inward.
            Space: O(1), because no normalized copy is built.
        """
        left = 0
        right = len(s) - 1

        while left < right:
            # Move `left` to the next character that participates in the
            # palindrome check. The boundary guard matters for inputs such as
            # "   !!!", where the normalized string is empty.
            while left < right and not s[left].isalnum():
                left += 1

            # Move `right` to the previous participating character.
            while left < right and not s[right].isalnum():
                right -= 1

            # Now s[left] and s[right] are the next mirrored characters in the
            # normalized string. A mismatch cannot be fixed by later choices.
            if s[left].lower() != s[right].lower():
                return False

            # This pair is proven. Shrink the unverified window.
            left += 1
            right -= 1

        # If the pointers cross, every required mirrored pair matched. This also
        # handles normalized strings of length 0 or 1.
        return True
