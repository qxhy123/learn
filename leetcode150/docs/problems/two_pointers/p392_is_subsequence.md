# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Tags: two-pointers, strings

## Core Pattern

When one ordered sequence must appear inside another, scan the larger sequence once and advance the smaller sequence only when the next required item matches. The smaller pointer always represents the next character you still need.

## Why Two Pointers Fits

A subsequence preserves order but not contiguity. That means `s` can skip characters from `t`, but it cannot reorder them. Two pointers fit because one pointer tracks the next needed character in `s`, while the other walks through `t` and offers candidates in the only order they can appear.

## Recommended Approach

1. Set `s_index = 0` to mean “the next needed character is `s[0]`.”
2. Scan `t` from left to right.
3. If `s_index == len(s)`, return `True` early because the whole subsequence is already matched.
4. When `t_char == s[s_index]`, advance `s_index` by one.
5. Otherwise ignore `t_char` and keep scanning.
6. After the loop ends, return whether `s_index == len(s)`.

The empty string is always a subsequence because there are no required characters to match.

## Alternative Approaches

If you need to answer many subsequence queries against the same `t`, you can preprocess the positions of each character and binary-search the next valid occurrence for every character in `s`. That is useful for repeated queries, but it is unnecessary for a single check. Recursive matching also expresses the same idea, but it adds call-stack overhead without improving the logic.

## Correctness Sketch

Maintain this invariant: `s[:s_index]` has already been matched as a subsequence of the prefix of `t` that has been scanned. Skipping a non-matching character in `t` is safe because subsequences are allowed to drop characters. Consuming a matching character is also safe because it is the next required character in order, and taking it cannot make a valid solution disappear. When the scan ends, `s` is a subsequence of `t` exactly when every required character has been consumed, which is equivalent to `s_index == len(s)`.

Repeated characters are handled naturally by the same invariant. If `s` needs two `a` characters, the pointer in `s` advances only after two distinct matching positions have been seen in `t` in the correct order.

## Trace

For `s = "abc"` and `t = "ahbgdc"`:

| `t` character | `s_index` before | Needed char | Action |
| --- | --- | --- | --- |
| `a` | `0` | `a` | Match, advance to need `b` |
| `h` | `1` | `b` | Skip |
| `b` | `1` | `b` | Match, advance to need `c` |
| `g` | `2` | `c` | Skip |
| `d` | `2` | `c` | Skip |
| `c` | `2` | `c` | Match, subsequence complete |

For `s = "aab"` and `t = "aaab"`, the repeated `a` characters in `t` are enough because the pointer in `s` only advances after each ordered match. If `t` had only one `a`, the second `a` requirement would remain unmet and the answer would be `False`.

## Complexity

- Time: `O(len(t))` for one query because each character in `t` is inspected once.
- Space: `O(1)` because the algorithm keeps only a pointer into `s`.

## Common Pitfalls

- Confusing subsequence with substring and requiring contiguity.
- Advancing `s_index` on a mismatch instead of waiting for the next matching character.
- Forgetting that the empty string is always a subsequence.
- Assuming repeated characters in `t` are enough without checking order.
- Missing the early exit when `s_index` reaches `len(s)`.

## Implementation Notes

See `solutions/two_pointers/p392_is_subsequence.py`. The implementation uses one explicit pointer into `s`, and the loop over `t` acts as the second pointer. The early return once `s_index == len(s)` keeps the scan short when the subsequence is already complete.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`. The tests cover the official examples, an empty `s`, an empty `t`, repeated characters that require multiple matches, and false cases where the characters appear in the wrong order.

## Interview Script

"I scan `t` once while keeping a pointer to the next character I still need from `s`. Matching characters advance that pointer, and non-matching characters are skipped because subsequences can ignore them. If the pointer reaches the end of `s`, every required character appeared in order. Repeated characters are fine as long as `t` contains enough ordered matches."

## Review Questions

1. Why does skipping a non-matching character in `t` never hurt the answer?
2. Why is the empty string always a subsequence?
3. What does `s_index` mean at any point in the scan?
4. How do repeated characters change the reasoning, if at all?
5. When would a preprocessing approach be better than one linear scan?

## Follow-up Practice

- Count how many words in a list are subsequences of one string.
- Preprocess one fixed `t` to support many subsequence queries efficiently.
- Compare subsequence matching with substring matching to see why contiguity changes the problem.
