# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use symmetric pointers when the required property is defined by matching the left side of a sequence with the right side. Each iteration should either discard irrelevant input or prove that the current mirrored pair is valid before shrinking the window.

## Why Two Pointers Fits

Palindrome checking is naturally a two-ended problem: the first meaningful character must match the last meaningful character, then the second meaningful character must match the second-to-last, and so on. The input also contains characters that do not participate in the comparison. Two pointers let us skip those characters in place and compare only the meaningful mirrored pairs without allocating a filtered copy.

This is the simplest form of the two-pointer pattern: both pointers move inward, and the active window always represents the part of the string that still needs proof.

## Recommended Approach

1. Initialize `left` at index `0` and `right` at `len(s) - 1`.
2. While `left < right`, advance `left` until it points to an alphanumeric character or crosses `right`.
3. Similarly, move `right` backward until it points to an alphanumeric character or crosses `left`.
4. Compare `s[left].lower()` with `s[right].lower()`.
5. If they differ, return `False`; one required mirrored pair is invalid.
6. If they match, move both pointers inward and continue.
7. If the pointers meet or cross, every meaningful mirrored pair has matched, so return `True`.

## Alternative Approaches

The most direct alternative is to build a cleaned lowercase string and compare it with its reverse. That version is concise, but it uses `O(n)` extra space and hides the pointer invariant. The in-place scan is better for interviews because it exposes the reasoning: ignore irrelevant characters, compare the next required pair, and shrink the unresolved window.

A recursive mirrored comparison is also possible, but it adds call-stack overhead without improving clarity or asymptotic complexity.

## Correctness Sketch

Maintain this invariant: before each comparison, every meaningful character pair outside the current `[left, right]` window has already been checked and matched. The skip loops preserve the invariant because non-alphanumeric characters are irrelevant by definition. When both pointers reference meaningful characters, those two positions are the next required mirrored pair. If they differ after case normalization, no later decision can repair that mismatch, so returning `False` is correct. If they match, moving both pointers inward marks that pair as proven and restores the invariant for the smaller window. When the loop ends, no unproven mirrored pair remains, so the string is a valid palindrome.

## Trace

For `"A man, a plan, a canal: Panama"`:

| Window focus | Left action | Right action | Result |
| --- | --- | --- | --- |
| Full string | `A` is meaningful | `a` is meaningful | `a == a`, shrink |
| After shrinking | skip spaces and punctuation | `m` is meaningful | compare next meaningful pair |
| Middle scan | letters match in mirrored order | punctuation is skipped | invariant remains true |
| End | pointers meet/cross | no mismatch found | return `True` |

For `"0P"`, both characters are meaningful and `"0" != "p"`, so the algorithm returns `False` immediately.

## Complexity

- Time: `O(n)` because each pointer only moves inward and each character is inspected at most a constant number of times.
- Space: `O(1)` because the algorithm stores only two indices and temporary character comparisons.

## Common Pitfalls

- Skipping only spaces and forgetting punctuation or symbols.
- Comparing uppercase and lowercase letters directly.
- Accessing `s[left]` or `s[right]` after a skip loop without checking `left < right`.
- Building a filtered string when the intended follow-up asks for constant extra space.
- Treating an empty or punctuation-only string as false; after filtering, it has no mismatched pair, so it is a palindrome.

## Implementation Notes

See `solutions/two_pointers/p125_valid_palindrome.py`. The implementation keeps the skip loops inside the main `left < right` loop and uses `str.isalnum()` plus `str.lower()` to match the problem's comparison rules.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`. The tests cover official examples, punctuation-only input, mixed case, digits, and real mismatches after filtering.

## Interview Script

"I use two pointers because palindrome validity is symmetric. I move the left pointer to the next alphanumeric character and the right pointer to the previous alphanumeric character, compare them case-insensitively, and shrink inward. A mismatch proves failure; if the pointers cross, all required mirrored pairs matched."

## Review Questions

1. What exact invariant is true after each successful comparison?
2. Why is a punctuation-only string considered a valid palindrome?
3. Why is it safe to skip non-alphanumeric characters before comparing?
4. How would the space complexity change if we built a filtered string first?
5. What bug can occur if the skip loops do not check `left < right`?

## Follow-up Practice

- Valid Palindrome II: allow deleting at most one character.
- Compare two strings after applying backspaces.
- Check whether an array segment is symmetric under a custom equality rule.
