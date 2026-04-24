from __future__ import annotations

"""Valid Palindrome — two-pointer tutorial implementation.

The input may contain spaces, punctuation, digits, and mixed-case letters. The
palindrome decision is made only on alphanumeric characters, compared
case-insensitively. Instead of building a cleaned copy of the string, this
solution performs that filtering lazily from both ends.

Pattern:
    Use two inward-moving pointers when a property is defined by mirrored
    positions. Each pointer skips values that do not participate in the
    comparison, then the algorithm validates the next meaningful pair.

Invariant:
    Before every comparison, all meaningful character pairs outside the current
    [left, right] window have already matched. If the current meaningful pair
    differs, no later pointer movement can repair that mismatch.

Complexity:
    Time: O(n), because each character is crossed by at most one pointer.
    Space: O(1), because no filtered copy of the string is created.
"""


class Solution:
    """See `docs/problems/two_pointers/p125_valid_palindrome.md`."""

    def isPalindrome(self, s: str) -> bool:
        """Return whether `s` is a palindrome after normalization.

        Normalization rules follow the LeetCode problem statement:
        - ignore non-alphanumeric characters;
        - compare remaining characters case-insensitively.

        Example:
            "A man, a plan, a canal: Panama" is treated as
            "amanaplanacanalpanama", so the method returns True.

        The method intentionally scans the original string directly. A common
        alternative is `cleaned == cleaned[::-1]`, but that uses O(n) extra
        memory. The two-pointer version keeps the same linear runtime while
        preserving constant auxiliary space.
        """
        left = 0
        right = len(s) - 1

        while left < right:
            # Move both pointers to the next characters that actually matter.
            # The `left < right` guard avoids reading a crossed pointer on
            # inputs that are empty after filtering, such as ".,,   :;".
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1

            # These two characters are the next required mirrored pair in the
            # normalized string. Any mismatch proves the whole string invalid.
            if s[left].lower() != s[right].lower():
                return False

            # The pair matched, so shrink the unresolved window.
            left += 1
            right -= 1

        # No mismatched meaningful pair remains. This also covers strings that
        # normalize to the empty string or a single character.
        return True
