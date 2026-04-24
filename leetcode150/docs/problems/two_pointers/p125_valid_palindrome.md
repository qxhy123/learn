# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use symmetric pointers when a condition must hold between the beginning and the end of a sequence. Move each pointer inward after the current pair has been validated.

## Why Two Pointers Fits

A palindrome is defined by mirrored positions. After ignoring non-alphanumeric characters and normalizing case, the first meaningful character must match the last meaningful character, the second must match the second-to-last, and so on. Two pointers let us check those mirrored pairs without building a filtered copy of the string.

## Recommended Approach

1. Put `left` at the start of `s` and `right` at the end.
2. Move `left` forward while it points at a non-alphanumeric character.
3. Move `right` backward while it points at a non-alphanumeric character.
4. Compare the lowercase forms of the two meaningful characters.
5. If they differ, return `False` immediately.
6. Otherwise move both pointers inward and continue until they cross.
7. If every mirrored pair matches, return `True`.

## Alternative Approaches

A simple alternative is to build a filtered lowercase list and compare it with its reverse. That is easier to read but uses `O(n)` extra space. The in-place two-pointer scan is the interview-preferred version because it keeps the same linear time while using constant auxiliary space.

## Correctness Sketch

Maintain the invariant that every meaningful character pair outside the current `[left, right]` window has already been checked and matched. The skip loops remove characters that the problem says should not affect the palindrome decision. When both pointers reference meaningful characters, a mismatch proves no valid palindrome exists because those mirrored positions must be equal. If they match, shrinking the window preserves the invariant. When the pointers cross, every required mirrored pair has matched, so the string is a valid palindrome.

## Trace

For `"A man, a plan, a canal: Panama"`:

| Step | Left char | Right char | Action |
| --- | --- | --- | --- |
| 1 | `A` | `a` | Compare lowercase `a == a`, move inward |
| 2 | skip spaces/punctuation | `m` | Skip ignored characters until both sides are alphanumeric |
| 3 | `m` | `m` | Match and move inward |
| 4 | later pairs | later pairs | Continue matching mirrored letters |
| End | pointers cross |  | All meaningful pairs matched |

## Complexity

- Time: `O(n)` because each pointer moves across the string at most once.
- Space: `O(1)` because the algorithm stores only pointer indices.

## Common Pitfalls

- Filtering only spaces but not punctuation.
- Comparing characters without lowercasing them.
- Moving a pointer past the other pointer inside a skip loop.
- Allocating a filtered string when the interviewer asks for constant extra space.

## Implementation Notes

See `solutions/two_pointers/p125_valid_palindrome.py`. The key detail is that each skip loop checks `left < right` before reading the character.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`. The tests cover official examples, strings that become empty after filtering, mixed case, digits, punctuation, and mismatch detection after filtering.

## Interview Script

"I use two pointers because palindrome validity is symmetric. I skip non-alphanumeric characters on both sides, compare lowercase meaningful characters, and move inward. A mismatch immediately proves the string is not a palindrome; if the pointers cross, every mirrored pair matched."

## Review Questions

1. Why is it safe to ignore non-alphanumeric characters before comparing?
2. What invariant is preserved after each successful comparison?
3. Why does this solution use `O(1)` extra space?
4. What happens when the string contains no alphanumeric characters?

## Follow-up Practice

- Check if a string can become a palindrome after deleting at most one character.
- Validate palindromes in Unicode-aware text where normalization rules matter.
- Apply the same symmetric scan idea to arrays instead of strings.
