# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## First-Principles Explanation

### What The Problem Is Asking

Given two strings `s` and `t`, decide whether `s` can be found inside `t` as a subsequence.

A subsequence does **not** need to occupy consecutive positions. It only needs to preserve order. For example, `"abc"` is a subsequence of `"ahbgdc"` because we can choose:

```text
t: a h b g d c
   ^   ^     ^
s: a   b     c
```

The chosen characters appear in the same left-to-right order as `s`. The extra characters `h`, `g`, and `d` are simply skipped.

By contrast, `"axc"` is not a subsequence of `"ahbgdc"`. We can match `a`, but there is no `x` after that `a`, so the required order cannot be completed.

The problem is therefore not asking whether `s` is a substring, not asking whether the two strings contain the same characters, and not asking whether every character of `s` appears somewhere independently. It asks a stricter ordered question:

> Can we assign each character of `s` to a distinct position in `t` so that those positions strictly increase?

### Brute-Force Baseline

A direct brute-force way to think about the problem is: try every possible set of positions in `t` whose length is `len(s)`, then check whether those chosen positions spell `s`.

For `s = "abc"` and `t = "ahbgdc"`, we could try combinations such as:

```text
positions 0,1,2 -> "ahb"
positions 0,2,5 -> "abc"  match
positions 1,3,4 -> "hgd"
...
```

This captures the definition correctly, but it is much too expensive. If `t` has length `n` and `s` has length `m`, there can be many ways to choose `m` positions from `n` positions. Most of that work is unnecessary because once a character of `s` is matched, earlier positions in `t` can never help match later characters of `s`.

A slightly better baseline is recursive search: for each character of `s`, scan forward in `t` and try every matching occurrence. That still branches when a character appears many times. For example, matching many `a` characters inside a target with many `a` characters creates many equivalent choices.

The key question is: do we really need to try all matching occurrences, or is one choice always safe?

### Key Observation

When matching a subsequence from left to right, the earliest possible match for the current character is always at least as good as any later match.

Suppose the next needed character is `s[i]`, and while scanning `t` we find it at position `j`. If we choose this earliest `j`, then every later position in `t` remains available for `s[i + 1:]`. If instead we skip this match and choose the same character at some later position `k > j`, we have thrown away positions between `j + 1` and `k - 1` for no benefit.

Choosing the earliest match leaves the largest possible suffix of `t` for the remaining characters. That makes the greedy local choice safe.

This observation turns the problem into a single left-to-right scan:

- Keep track of the next character of `s` that still needs to be matched.
- Walk through `t` from left to right.
- Whenever the current character of `t` equals that next needed character, consume it and move to the next character of `s`.
- If all characters of `s` are consumed, `s` is a subsequence.

### Two-Pointer Subsequence Invariant

The two pointers represent different roles:

- `s_index`: the first unmatched character in `s`.
- `t_index`: the current character being inspected in `t`.

The implementation may write `t_index` as an explicit integer, or it may use a `for char in t` loop. Conceptually, the loop over `t` is the second pointer.

The invariant is:

> Before each step, `s[:s_index]` has already been matched as a subsequence of the portion of `t` that has been scanned, and `s_index` is the next character of `s` that still needs a match.

This invariant is the whole reason the algorithm is simple.

When the current `t` character does not equal `s[s_index]`, it cannot help with the next required character. Since subsequence order is fixed, we are not allowed to use it for a later character of `s` before matching the current one. So we skip it.

When the current `t` character equals `s[s_index]`, matching it is safe because it is the earliest available match for that required character. We advance `s_index`, and the invariant remains true for one more matched character.

### Detailed Algorithm

1. Start `s_index = 0`, meaning no characters of `s` have been matched yet.
2. Scan each character `char` of `t` from left to right.
3. If `s_index == len(s)`, every character of `s` has already been matched, so return `True`.
4. Compare `char` with `s[s_index]`, the next required character.
5. If they are equal, advance `s_index` by one.
6. If they are not equal, ignore `char` and continue scanning `t`.
7. After the scan finishes, return whether `s_index == len(s)`.

The empty string case falls out naturally. If `s` is empty, then `len(s) == 0`, so zero characters need to be matched. The answer is `True`, even when `t` is also empty.

### Example Walkthrough: `s = "abc"`, `t = "ahbgdc"`

Start with `s_index = 0`, so the next needed character is `s[0] = 'a'`.

```text
s = a b c
    ^
t = a h b g d c
```

Scan `t`:

1. `t[0] = 'a'`, next needed is `'a'`. Match it. Now `s_index = 1`, next needed is `'b'`.
2. `t[1] = 'h'`, next needed is `'b'`. Skip `h`.
3. `t[2] = 'b'`, next needed is `'b'`. Match it. Now `s_index = 2`, next needed is `'c'`.
4. `t[3] = 'g'`, next needed is `'c'`. Skip `g`.
5. `t[4] = 'd'`, next needed is `'c'`. Skip `d`.
6. `t[5] = 'c'`, next needed is `'c'`. Match it. Now `s_index = 3`.

Since `s_index == len(s)`, all characters of `s` were matched in order. Return `True`.

### Example Walkthrough: `s = "axc"`, `t = "ahbgdc"`

```text
s = a x c
t = a h b g d c
```

1. Match `a` at `t[0]`. Now the next needed character is `x`.
2. Scan the rest of `t`: `h`, `b`, `g`, `d`, `c`.
3. None of them is `x`.

The scan ends while `s_index` still points at `x`, so not all of `s` was matched. Return `False`.

Notice that the final `c` in `t` does not help. The problem requires `x` before `c`; we cannot skip an unmatched required character of `s` and come back to it later.

### Code

```python
def isSubsequence(s: str, t: str) -> bool:
    s_index = 0

    for char in t:
        if s_index == len(s):
            return True
        if s[s_index] == char:
            s_index += 1

    return s_index == len(s)
```

Equivalent pseudocode with both pointers shown explicitly:

```text
s_index = 0
t_index = 0

while s_index < len(s) and t_index < len(t):
    if s[s_index] == t[t_index]:
        s_index += 1
    t_index += 1

return s_index == len(s)
```

The Python implementation uses a `for` loop for `t`, so `t_index` is implicit. The behavior is the same: every character of `t` is inspected at most once, and `s_index` advances only when the next required character is matched.

### Correctness

We prove that the algorithm returns `True` exactly when `s` is a subsequence of `t`.

First, the invariant holds at the start. Before scanning any character of `t`, `s_index = 0`, and the empty prefix `s[:0]` has been matched in the scanned portion of `t`.

Now consider one scanned character from `t`.

- If it does not equal `s[s_index]`, skipping it is safe. It cannot match the next required character. Because subsequence matching must preserve order, a character that cannot match the next requirement cannot be used for a later requirement yet. The already matched prefix of `s` is unchanged, so the invariant remains true.
- If it equals `s[s_index]`, the algorithm uses it to match the next required character. The previous matched prefix was already a subsequence of the earlier scanned characters, and this matching character appears after them, so `s[:s_index + 1]` is now matched in order. Advancing `s_index` preserves the invariant.

By induction, after every scanned character, `s[:s_index]` is exactly a prefix of `s` that has been matched in order inside the scanned part of `t`.

If the algorithm returns `True`, then `s_index == len(s)`. The invariant says `s[:len(s)]`, which is all of `s`, has been matched as a subsequence of `t`. Therefore `s` is a subsequence.

If the algorithm returns `False`, the scan of `t` has ended and `s_index < len(s)`. The greedy choice always took the earliest available match for each required character. Taking an earliest match leaves at least as much remaining `t` as any later choice would have left. Therefore, if the algorithm could not match the next required character after making these safest possible choices, no alternative set of positions can complete the subsequence. Therefore `s` is not a subsequence.

### Complexity

- Time: `O(len(t))` for the scan. Since `s_index` only moves forward and never exceeds `len(s)`, this is also `O(len(s) + len(t))` if both input sizes are counted explicitly.
- Space: `O(1)` auxiliary space. The algorithm stores only an index and loop state.

### Common Pitfalls

- Confusing subsequence with substring. A subsequence may skip characters; a substring must be contiguous.
- Checking character counts only. `"aec"` and `"abcde"` share the needed letters, but `e` appears after `c`, so `"aec"` is not a subsequence.
- Advancing the `s` pointer on every character of `t`. The `s` pointer should advance only when the next required character is matched.
- Resetting the scan of `t` for each character of `s`. That destroys the order constraint and can incorrectly reuse earlier positions.
- Forgetting the empty `s` case. The empty string is a subsequence of every string.
- Treating repeated characters as one match. `"aaaa"` is not a subsequence of `"baaac"` because there are only three `a` characters after scanning in order.

### First-Principles Summary

The definition of a subsequence is an ordered assignment from characters of `s` to increasing positions in `t`. Because positions must increase, the only useful direction is left to right. At any moment, the algorithm needs exactly one piece of state: how much of `s` has already been matched. When the current character of `t` matches the next needed character of `s`, using it immediately is optimal because it leaves the maximum possible remaining suffix of `t` for the rest of `s`. When it does not match, it cannot help with the next required character and can be skipped.

That is the two-pointer idea here: one pointer tracks progress through the pattern `s`, while the other pointer scans the candidate source `t`. The invariant says the matched prefix of `s` is always valid inside the scanned prefix of `t`. Once the matched prefix becomes all of `s`, the answer is `True`; if `t` runs out first, the answer is `False`.

## Implementation

See `solutions/two_pointers/p392_is_subsequence.py`.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`.

## Examples

- `s = "abc"`, `t = "ahbgdc"` returns `True` because `a`, `b`, and `c` can be matched in order while skipping other characters.
- `s = "axc"`, `t = "ahbgdc"` returns `False` because after matching `a`, no `x` appears before the scan ends.
- `s = ""`, `t = "anything"` returns `True` because zero required characters are already matched.
- `s = "aaaa"`, `t = "baaac"` returns `False` because only three `a` characters are available in order.
- See `tests/two_pointers/test_p392_is_subsequence.py` for executable examples and edge cases.

## Follow-up Practice

- Trace `s_index` for `s = "ace"`, `t = "abcde"` and identify each skipped character.
- Trace why `s = "aec"`, `t = "abcde"` fails even though all three letters appear in `t`.
- Explain why choosing the earliest possible match can never make a future match harder than choosing a later occurrence.
