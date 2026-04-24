# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Tags: two-pointers, strings

## Core Pattern

Use mirrored pointers when the answer depends on whether the left side and the right side agree after ignoring irrelevant symbols. Check one pair at a time, move inward, and keep the window shrinking until the pointers cross.

## Why Two Pointers Fits

This problem is a mirror test, not a full-string transformation problem. The first meaningful character must match the last meaningful character, the second must match the second-to-last, and so on. Two pointers let you verify those mirrored pairs directly while skipping punctuation and spaces on the fly, so you do not need to build a filtered copy first.

## Recommended Approach

1. Set `left = 0` and `right = len(s) - 1`.
2. Advance `left` while it points to a non-alphanumeric character and `left < right`.
3. Move `right` left while it points to a non-alphanumeric character and `left < right`.
4. Compare `s[left].lower()` with `s[right].lower()`.
5. If they differ, return `False` immediately.
6. Otherwise move both pointers inward and repeat.
7. If the loop finishes without a mismatch, return `True`.

An effectively empty string, such as `"   ,,, "`, is a palindrome because there are no meaningful characters to contradict the mirror rule.

## Alternative Approaches

A common alternative is to normalize the string first: keep only alphanumeric characters, convert them to lowercase, and compare the cleaned result with its reverse. That approach is easy to read, but it uses `O(n)` extra space for the filtered copy. The two-pointer version is the better pattern-transfer answer because it teaches the same mirror reasoning without the extra allocation.

## Correctness Sketch

Maintain this invariant: every meaningful character outside the current `[left, right]` window has already been matched with its mirrored partner. The skip loops are safe because punctuation and spaces do not affect the palindrome decision. If two meaningful characters differ after lowercase normalization, then the string cannot be a palindrome because mirrored positions must match exactly under the problem’s comparison rules. If the characters match, moving both pointers inward preserves the invariant. When the pointers cross, every mirrored meaningful pair has matched, so the string is a palindrome. If there are no meaningful characters at all, the invariant is vacuously true, so the answer is also `True`.

## Trace

For `"A man, a plan, a canal: Panama"`:

| Step | Left pointer | Right pointer | Meaningful pair | Action |
| --- | --- | --- | --- | --- |
| 1 | `A` | `a` | `a` vs `a` | Match, move both pointers inward |
| 2 | skip space | `m` | `m` vs `m` | Skip ignored characters, then match |
| 3 | `a` | `a` | `a` vs `a` | Match, continue inward |
| 4 | `p` | `p` | `p` vs `p` | Match, continue inward |
| End | pointers cross | pointers cross | all mirrored pairs checked | Return `True` |

For `"0P"`, the first meaningful pair is `0` and `p`, which does not match after lowercasing, so the algorithm returns `False` immediately.

## Complexity

- Time: `O(n)` because each pointer advances across the string at most once.
- Space: `O(1)` because the algorithm only stores indices and a few temporary characters.

## Common Pitfalls

- Skipping spaces but forgetting punctuation and symbols.
- Comparing raw characters without lowercase normalization.
- Reading `s[left]` or `s[right]` without guarding the skip loops with `left < right`.
- Building a filtered string when the interviewer asked for constant auxiliary space.
- Treating an effectively empty string as a special failure case instead of a valid palindrome.

## Implementation Notes

See `solutions/two_pointers/p125_valid_palindrome.py`. The implementation uses `str.isalnum()` to decide what to skip and `str.lower()` to compare the mirrored characters in a case-insensitive way.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`. The tests cover the official examples, strings that become empty after filtering, mixed case with digits, punctuation-heavy inputs, and a false case that only fails after non-alphanumeric characters are removed.

## Interview Script

"I use two pointers because palindrome checking is symmetric. I skip non-alphanumeric characters on both sides, compare the lowercase meaningful characters, and move inward. If any mirrored pair disagrees, the string is not a palindrome. If the pointers cross, every required pair matched, and even an effectively empty string counts as a palindrome."

## Review Questions

1. Why can punctuation and spaces be ignored without changing the answer?
2. What does the loop invariant say about the window that remains unexplored?
3. Why does an effectively empty string return `True`?
4. Why is lowercase normalization required before comparison?
5. What makes the scan `O(1)` extra space instead of `O(n)`?

## Follow-up Practice

- Check whether a string can become a palindrome after deleting at most one character.
- Adapt the same mirrored scan to arrays of numbers instead of characters.
- Handle palindrome checking with Unicode normalization rules instead of ASCII-style character classes.
