# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When one sequence must appear inside another while preserving order, scan the larger sequence once and advance the smaller-sequence pointer only when a match is found.

## Why Two Pointers Fits

A subsequence does not require contiguous positions, but it does require relative order. That means each character in `s` must be matched by a later character in `t`. A pointer into `s` tracks the next required character, while the scan through `t` offers candidates in order.

## Recommended Approach

1. Start `s_index` at `0`, meaning the next needed character is `s[0]`.
2. Iterate through every character in `t` from left to right.
3. If all characters in `s` have already matched, return `True`.
4. When the current `t` character equals `s[s_index]`, advance `s_index`.
5. Ignore non-matching `t` characters because subsequences may skip characters.
6. At the end, return whether `s_index == len(s)`.

## Alternative Approaches

A recursive solution can express the same idea but uses call-stack space and is unnecessary. For many repeated queries against the same `t`, a preprocessing approach can store character positions and binary-search the next valid position for each character of `s`. For one query, the linear two-pointer scan is simpler and optimal.

## Correctness Sketch

Maintain the invariant that `s[:s_index]` has been matched as a subsequence of the part of `t` already scanned. If the next `t` character does not match `s[s_index]`, skipping it cannot hurt because a subsequence is allowed to ignore characters. If it matches, consuming it is safe because it is the earliest available match for the next required character. When the scan ends, all of `s` is a subsequence exactly when every required character has been consumed.

## Trace

For `s = "abc"`, `t = "ahbgdc"`:

| `t` char | Needed char | Action |
| --- | --- | --- |
| `a` | `a` | Match, advance to need `b` |
| `h` | `b` | Skip |
| `b` | `b` | Match, advance to need `c` |
| `g` | `c` | Skip |
| `d` | `c` | Skip |
| `c` | `c` | Match, all of `s` consumed |

## Complexity

- Time: `O(len(t))` for a single query because each character in `t` is scanned once.
- Space: `O(1)` because only the pointer into `s` is stored.

## Common Pitfalls

- Sorting either string, which destroys order information.
- Requiring characters to be contiguous, which would solve substring matching instead.
- Forgetting that the empty string is a subsequence of every string.
- Advancing the `s` pointer on a mismatch.

## Implementation Notes

See `solutions/two_pointers/p392_is_subsequence.py`. The implementation uses one explicit pointer for `s`; the loop over `t` acts as the second pointer.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`. The tests cover official examples, empty strings, repeated characters, insufficient repeated matches, and order-sensitive false cases.

## Interview Script

"I scan the target string once while keeping a pointer to the next character needed from the subsequence. A mismatch is ignored, and a match advances the subsequence pointer. If that pointer reaches the end of `s`, every character appeared in order."

## Review Questions

1. Why does skipping a non-matching character in `t` never remove a needed solution?
2. Why is the empty string always a subsequence?
3. How would the approach change for many different `s` queries against one fixed `t`?
4. What invariant does `s_index` represent?

## Follow-up Practice

- Preprocess `t` for many subsequence queries.
- Count how many words in a list are subsequences of one string.
- Compare subsequence matching with substring matching.
