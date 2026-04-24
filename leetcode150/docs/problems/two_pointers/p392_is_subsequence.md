# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use a requirement pointer and a scan pointer when one sequence must be found inside another in order. The scan pointer explores candidates; the requirement pointer advances only when the next required item is satisfied.

## Why Two Pointers Fits

A subsequence is order-preserving but not necessarily contiguous. That means characters in `t` are allowed to be skipped, but characters in `s` must be matched in their original order. This creates a natural two-pointer interpretation:

- one pointer marks the next character of `s` that still needs a match;
- the other pointer scans `t` from left to right, offering candidates in legal order.

The official constraints make `s` short and `t` potentially much longer. A single linear scan of `t` is exactly what we want for one query.

## Recommended Approach

1. Set `s_index = 0`; this points to the next unmatched character of `s`.
2. Iterate through each character `char` in `t`.
3. If `s_index == len(s)`, return `True` because all requirements are already matched.
4. If `char == s[s_index]`, consume that requirement by incrementing `s_index`.
5. If not, ignore `char`; subsequences may skip characters.
6. After scanning `t`, return whether `s_index == len(s)`.

## Alternative Approaches

A recursive version can try to match or skip characters, but it is unnecessary for a single subsequence query and risks extra stack usage. For many queries against the same `t`, a stronger approach is to preprocess `t` into sorted position lists for each character, then binary-search the next usable position for each character in `s`. That follow-up trades preprocessing and extra memory for faster repeated queries.

For this problem, the direct scan is the right tool: it is simple, linear, and uses constant space.

## Correctness Sketch

Maintain this invariant: after scanning some prefix of `t`, `s[:s_index]` is matched as a subsequence of that scanned prefix, and no longer prefix of `s` has been matched.

If the current character in `t` does not equal `s[s_index]`, it cannot satisfy the next requirement, so skipping it does not lose a valid match. If it does equal `s[s_index]`, taking it is safe because it is the earliest possible match for that requirement, leaving the rest of `t` available for later requirements. Therefore `s_index` always tracks exactly how much of `s` has been matched. At the end, `s` is a subsequence if and only if `s_index` reached `len(s)`.

## Trace

For `s = "abc"` and `t = "ahbgdc"`:

| Character from `t` | Needed from `s` | Action | Matched prefix |
| --- | --- | --- | --- |
| `a` | `a` | match and advance | `a` |
| `h` | `b` | skip | `a` |
| `b` | `b` | match and advance | `ab` |
| `g` | `c` | skip | `ab` |
| `d` | `c` | skip | `ab` |
| `c` | `c` | match and advance | `abc` |

Since the whole `s` has been consumed, the answer is `True`.

## Complexity

- Time: `O(len(t))` for one query. Each character of `t` is inspected once.
- Space: `O(1)`. Only the pointer into `s` is stored.

## Common Pitfalls

- Confusing subsequence with substring and requiring contiguous matches.
- Advancing the `s` pointer when the current `t` character does not match.
- Forgetting that empty `s` is always a subsequence.
- Sorting the strings, which destroys the required order.
- Mishandling repeated characters, where each occurrence in `s` needs a distinct later occurrence in `t`.

## Implementation Notes

See `solutions/two_pointers/p392_is_subsequence.py`. The implementation uses the loop over `t` as the scan pointer and `s_index` as the requirement pointer. It returns early once every character of `s` is matched.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`. The tests cover official true/false cases, empty inputs, repeated characters, insufficient repeated matches, and order-sensitive failures.

## Interview Script

"I scan `t` once while keeping a pointer to the next character I need from `s`. If the current character in `t` matches that need, I advance the `s` pointer; otherwise I skip it. If the `s` pointer reaches the end, all characters appeared in order."

## Review Questions

1. Why can unmatched characters in `t` be skipped greedily?
2. What does `s_index` represent after scanning the first `k` characters of `t`?
3. Why does repeated-character input require careful testing?
4. How would you optimize many subsequence queries against one fixed `t`?
5. Why does sorting break the problem?

## Follow-up Practice

- Number of Matching Subsequences.
- Preprocess a string for repeated subsequence queries.
- Longest Common Subsequence, where both sequences have choices.
