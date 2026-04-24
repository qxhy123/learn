# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use two inward-moving pointers when the decision depends on mirrored positions. At every step, make the two pointers land on the next meaningful pair, prove that pair is compatible, and then shrink the still-unverified window.

This is the cleanest version of the two-pointer idea: the left pointer represents the earliest character that still needs a mirror, and the right pointer represents its latest possible mirror.

## Why Two Pointers Fits

A palindrome is not about every raw character in the input; it is about the sequence that remains after ignoring non-alphanumeric characters and normalizing case. Once that normalized sequence is imagined, the first element must equal the last, the second must equal the second-to-last, and so on.

Two pointers fit because we do not need to materialize that normalized sequence. The left pointer can skip irrelevant characters from the front, the right pointer can skip irrelevant characters from the back, and the first real disagreement proves the answer is false. If no disagreement appears before the pointers meet, every required mirrored pair has been verified.

The official constraints allow strings up to `2 * 10^5`, so avoiding an extra filtered copy is a meaningful improvement in space usage.

## Recommended Approach

1. Set `left = 0` and `right = len(s) - 1`.
2. While `left < right`, move `left` forward until it reaches an alphanumeric character or crosses `right`.
3. Move `right` backward until it reaches an alphanumeric character or crosses `left`.
4. If both pointers still describe a pair, compare `s[left].lower()` with `s[right].lower()`.
5. Return `False` immediately on a mismatch.
6. On a match, increment `left` and decrement `right` because that mirrored pair is now proven.
7. Return `True` after the loop ends.

## Alternative Approaches

The most readable baseline is:

1. Build a lowercase list of alphanumeric characters.
2. Compare that list with its reverse.

That baseline is useful for explaining the problem, but it uses `O(n)` extra space. The two-pointer implementation performs the same logical filtering lazily from both ends and uses `O(1)` extra space.

A recursive mirror check is another possibility, but it adds call-stack usage and makes skipping ignored characters more awkward. For interviews, the iterative two-pointer version is both simpler and more robust.

## Correctness Sketch

Maintain this invariant: before each comparison, every meaningful mirrored pair outside the current `[left, right]` interval has already been checked and matched.

The skip loops preserve the invariant because ignored characters are not part of the normalized palindrome sequence. When the loops stop, `s[left]` and `s[right]` are the next meaningful characters that must mirror each other. If their lowercase forms differ, the normalized sequence has a mismatched mirrored pair, so no valid palindrome exists. If they match, removing both from the unresolved interval preserves the invariant. When the pointers cross, there are no unresolved mirrored pairs left, so the normalized sequence reads the same forward and backward.

## Trace

For `s = "A man, a plan, a canal: Panama"`:

| Phase | `left` focus | `right` focus | Decision |
| --- | --- | --- | --- |
| Start | `A` | `a` | `a == a`, shrink |
| Skip ignored chars | spaces and comma are skipped | `m` | compare next meaningful pair |
| Middle | letters continue to mirror | punctuation is skipped | invariant remains true |
| Finish | pointers meet/cross | no mismatch | return `True` |

For `s = "0P"`, both characters are meaningful. The comparison is `"0"` vs `"p"`, so the algorithm returns `False` immediately.

## Complexity

- Time: `O(n)`, where `n` is `len(s)`. Each pointer moves in one direction and crosses each character at most once.
- Space: `O(1)`. The algorithm stores only two indices and temporary character comparisons.

## Common Pitfalls

- Treating spaces as the only ignored characters; punctuation and symbols must also be skipped.
- Forgetting case normalization before comparing letters.
- Letting a skip loop move a pointer past the other pointer and then reading out of the valid window.
- Returning `False` for a string that becomes empty after filtering; an empty normalized sequence is a palindrome.
- Allocating a filtered string when the interviewer explicitly asks for constant auxiliary space.

## Implementation Notes

See `solutions/two_pointers/p125_valid_palindrome.py`. The implementation uses `str.isalnum()` for the official alphanumeric rule and `str.lower()` for case-insensitive comparison. Both skip loops include `left < right`, which prevents crossing pointers from being dereferenced as a pair.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`. The tests cover official examples, strings that become empty after filtering, mixed punctuation/case, digits, and actual mismatches after filtering.

## Interview Script

"I treat the palindrome check as a mirrored-pair problem. I keep one pointer at the left and one at the right, skip characters that do not count, compare the lowercase meaningful characters, and shrink inward. The invariant is that everything outside the current window has already matched. A mismatch returns false; crossing pointers means all meaningful pairs matched."

## Review Questions

1. What is the unresolved window at any point in the algorithm?
2. Why does skipping punctuation preserve correctness?
3. Why is a punctuation-only string valid?
4. Where does the `O(1)` space improvement come from compared with filtering first?
5. What condition prevents pointer-crossing bugs inside the skip loops?

## Follow-up Practice

- Valid Palindrome II, where one deletion is allowed.
- Backspace String Compare, where ignored characters depend on previous input.
- Case-insensitive palindrome checking under Unicode normalization rules.
