# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use one pointer to track the next required item and another pointer to scan the source of candidates. Advance the requirement pointer only when the current candidate satisfies it.

## Why Two Pointers Fits

A subsequence preserves order but does not require contiguity. That means characters in `t` can be skipped freely, while characters in `s` must be consumed in exact order. The pointer into `s` represents the next unmatched requirement. The scan through `t` presents possible matches in the only order allowed by the problem.

This is a two-pointer pattern even though one pointer is expressed as a `for` loop: the loop index walks through `t`, and `s_index` walks through `s` only when a match is found.

## Recommended Approach

1. Initialize `s_index = 0`.
2. Iterate through `t` from left to right.
3. If `s_index == len(s)`, every required character has already matched; return `True`.
4. If the current character from `t` equals `s[s_index]`, advance `s_index`.
5. Otherwise, ignore the current character from `t`.
6. After the scan, return whether `s_index == len(s)`.

## Alternative Approaches

A recursive solution mirrors the definition of a subsequence, but it adds call-stack overhead and repeated branching. For many different `s` queries against the same `t`, preprocessing `t` into character positions and binary-searching the next valid position can be faster per query. For a single query, the linear scan is simpler, optimal, and easier to explain.

## Correctness Sketch

Maintain this invariant: after scanning a prefix of `t`, `s[:s_index]` is the longest prefix of `s` that can be matched as a subsequence of that scanned prefix. If the current `t` character does not equal the next needed character, skipping it cannot reduce the best possible match because it could not satisfy the current requirement. If it matches, consuming it is safe because it is the earliest available match for that requirement, leaving the maximum remaining suffix of `t` for later characters. At the end, all of `s` is a subsequence exactly when every required character has been consumed.

## Trace

For `s = "abc"`, `t = "ahbgdc"`:

| Scanned char in `t` | Next needed in `s` | Action | Matched prefix |
| --- | --- | --- | --- |
| `a` | `a` | match | `a` |
| `h` | `b` | skip | `a` |
| `b` | `b` | match | `ab` |
| `g` | `c` | skip | `ab` |
| `d` | `c` | skip | `ab` |
| `c` | `c` | match | `abc` |

The pointer reaches the end of `s`, so the answer is `True`.

## Complexity

- Time: `O(len(t))` for one query because the scan visits each character of `t` at most once.
- Space: `O(1)` because only the index into `s` is stored.

## Common Pitfalls

- Checking for substring containment instead of subsequence containment.
- Sorting either string, which destroys order.
- Returning `False` for empty `s`; the empty sequence is always a subsequence.
- Advancing the `s` pointer on mismatches.
- Forgetting that repeated characters need repeated ordered matches.

## Implementation Notes

See `solutions/two_pointers/p392_is_subsequence.py`. The implementation returns early once all of `s` is matched, but also works when `s` is empty because the final equality check succeeds.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`. The tests cover official examples, empty inputs, repeated characters, insufficient repeated matches, and order-sensitive false cases.

## Interview Script

"I keep a pointer to the next character I need from `s` and scan `t` once. When the current character in `t` matches that need, I advance the pointer in `s`; otherwise I skip it. If I consume all of `s`, then its characters appeared in order inside `t`."

## Review Questions

1. What does `s_index` mean at any point in the scan?
2. Why can a non-matching character in `t` be ignored safely?
3. Why does repeated-character input test the algorithm more strongly?
4. How would the design change for thousands of `s` queries against one `t`?
5. Why is subsequence matching different from substring matching?

## Follow-up Practice

- Number of Matching Subsequences.
- Preprocess a string for repeated subsequence queries.
- Longest Common Subsequence, where both sequences have branching choices.
