# 28. Find the Index of the First Occurrence in a String

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: string-matching

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two strings:

```text
haystack
needle
```

You need to return the smallest index `i` such that the substring of `haystack`
starting at `i` is exactly equal to `needle`.

In other words, find the first position where this is true:

```text
haystack[i : i + len(needle)] == needle
```

If no such position exists, return `-1`.

The word **first** matters. If `needle` appears more than once, the answer is
not any matching index. It is the leftmost matching index.

For example:

```text
haystack = "sadbutsad"
needle   = "sad"
```

There are two matches:

```text
haystack[0:3] = "sad"
haystack[6:9] = "sad"
```

The answer is `0`, because index `0` is the first occurrence.

The problem is not asking whether the characters of `needle` appear somewhere
in order. They must appear as one contiguous block.

For example:

```text
haystack = "saxbxd"
needle   = "sad"
```

The characters `s`, `a`, and `d` appear in order, but they are not contiguous.
So `"sad"` does not occur as a substring.

### 2. Name the Sizes

Let:

```text
n = len(haystack)
m = len(needle)
```

A match can only start at an index where there is enough room for all `m`
characters of `needle`.

So valid start positions are:

```text
0, 1, 2, ..., n - m
```

If `m > n`, then there is not enough room for `needle` anywhere in `haystack`,
so the answer must be `-1`.

Under the usual LeetCode constraints for this problem, `needle` is non-empty.
If this function were written for a more general API, the conventional answer
for an empty `needle` would be `0`, because the empty string occurs at the start
of every string.

### 3. Start From the Brute Force Baseline

The most literal solution is:

1. Try every possible start index in `haystack`.
2. Compare `needle` character by character against that position.
3. Return the first start index where every character matches.
4. If no start index works, return `-1`.

Pseudocode:

```python
for start in range(0, n - m + 1):
    matched = True

    for j in range(m):
        if haystack[start + j] != needle[j]:
            matched = False
            break

    if matched:
        return start

return -1
```

This is easy to reason about.

For a fixed `start`, the inner loop checks exactly the definition of a match:

```text
haystack[start + 0] == needle[0]
haystack[start + 1] == needle[1]
...
haystack[start + m - 1] == needle[m - 1]
```

Because the outer loop tries starts from left to right, the first successful
start is the required answer.

### 4. Why the Baseline Can Repeat Work

The brute force algorithm may compare the same parts of the strings many times.

Consider:

```text
haystack = "aaaaaaaaab"
needle   = "aaaab"
```

At start `0`, the first four characters match:

```text
haystack: a a a a a a a a a b
needle:   a a a a b
          0 1 2 3 4
```

Then the comparison fails at the last character:

```text
haystack[4] = 'a'
needle[4]   = 'b'
```

At start `1`, the algorithm again compares four `a` characters before failing.
At start `2`, it does the same again.

The repeated work comes from this fact:

> After a mismatch, the characters that already matched are not useless.
> They tell us something about where the next possible match could begin.

The optimized solution is built from that observation.

### 5. The Key Observation Behind KMP

Suppose we are matching `needle` against `haystack`, and we have already matched
some prefix of `needle`.

For example:

```text
needle = "ababaca"
```

Imagine we have matched this prefix:

```text
"ababa"
```

Then the next comparison fails.

The brute force algorithm would move the start by one and compare from the
beginning of `needle` again.

But we know the text we just matched:

```text
matched text = "ababa"
```

Some suffix of this matched text may also be a prefix of `needle`.

For `"ababa"`:

```text
suffix "aba" == prefix "aba"
```

So after the mismatch, we do not necessarily have to restart from `needle[0]`.
The last three matched characters may already represent the beginning of the
next candidate match.

This is the core KMP idea:

> When a mismatch happens after matching `k` characters, keep the longest suffix
> of those `k` characters that is also a prefix of `needle`.

That suffix can still be part of a future match. Everything before it can be
discarded safely.

### 6. Prefixes, Suffixes, and the LPS Table

KMP precomputes information about `needle` only.

For each position `i` in `needle`, we store:

```text
lps[i] = length of the longest proper prefix of needle[0:i + 1]
         that is also a suffix of needle[0:i + 1]
```

`lps` stands for "longest prefix suffix".

A **proper prefix** means the whole string itself is not allowed.

For example, for:

```text
needle = "ababaca"
```

Look at the prefix ending at index `4`:

```text
needle[0:5] = "ababa"
```

Its prefixes are:

```text
"a"
"ab"
"aba"
"abab"
```

Its suffixes are:

```text
"a"
"ba"
"aba"
"baba"
```

The longest proper prefix that is also a suffix is:

```text
"aba"
```

So:

```text
lps[4] = 3
```

This value means:

> If we have matched `"ababa"` and then fail on the next character, we can keep
> the last `3` matched characters because they already match the first `3`
> characters of `needle`.

### 7. Building the LPS Table From First Principles

We build `lps` from left to right.

State:

```text
i      = current index in needle whose lps value we are computing
length = length of the current candidate prefix-suffix match
```

Before computing `lps[i]`, `length` represents:

```text
the longest prefix of needle that is also a suffix of needle[0:i]
```

Now compare:

```text
needle[i] and needle[length]
```

There are two cases.

#### Case 1: The Characters Match

If:

```text
needle[i] == needle[length]
```

then the previous prefix-suffix match can be extended by one character.

So:

```text
length += 1
lps[i] = length
i += 1
```

#### Case 2: The Characters Do Not Match

If:

```text
needle[i] != needle[length]
```

then the current candidate prefix-suffix of size `length` is too large.

But there may be a smaller prefix-suffix candidate inside it.

The next best candidate length is:

```text
lps[length - 1]
```

Why?

Because if a suffix of the already matched candidate is also a prefix of
`needle`, that suffix is the only part that could still be extended by
`needle[i]`.

So we reduce:

```text
length = lps[length - 1]
```

Then we try the comparison again.

If `length` is already `0` and the characters still do not match, there is no
proper prefix-suffix ending at `i`.

So:

```text
lps[i] = 0
i += 1
```

### 8. LPS Walkthrough for `"ababaca"`

Let:

```text
needle = "ababaca"
index:    0 1 2 3 4 5 6
chars:    a b a b a c a
```

Initialize:

```text
lps[0] = 0
i = 1
length = 0
```

Index `1`, character `b`:

```text
needle[1] = 'b'
needle[0] = 'a'
```

They do not match, and `length` is `0`.

```text
lps[1] = 0
```

Index `2`, character `a`:

```text
needle[2] = 'a'
needle[0] = 'a'
```

They match.

```text
length = 1
lps[2] = 1
```

Index `3`, character `b`:

```text
needle[3] = 'b'
needle[1] = 'b'
```

They match.

```text
length = 2
lps[3] = 2
```

Index `4`, character `a`:

```text
needle[4] = 'a'
needle[2] = 'a'
```

They match.

```text
length = 3
lps[4] = 3
```

Index `5`, character `c`:

```text
needle[5] = 'c'
needle[3] = 'b'
```

Mismatch. Fall back:

```text
length = lps[2] = 1
```

Try again:

```text
needle[5] = 'c'
needle[1] = 'b'
```

Mismatch. Fall back:

```text
length = lps[0] = 0
```

Try again:

```text
needle[5] = 'c'
needle[0] = 'a'
```

Mismatch at length `0`.

```text
lps[5] = 0
```

Index `6`, character `a`:

```text
needle[6] = 'a'
needle[0] = 'a'
```

They match.

```text
length = 1
lps[6] = 1
```

Final table:

```text
needle: a b a b a c a
index:  0 1 2 3 4 5 6
lps:    0 0 1 2 3 0 1
```

### 9. Searching With the LPS Table

During the main scan, use two indices:

```text
i = index in haystack
j = index in needle
```

The state invariant is:

```text
needle[0:j] matches the last j characters of haystack that end just before i
```

Equivalently:

```text
haystack[i - j : i] == needle[0:j]
```

That means `j` is not just a pointer. It is the length of the currently matched
prefix of `needle`.

Now compare:

```text
haystack[i] and needle[j]
```

There are three important cases.

#### Case 1: The Characters Match

If:

```text
haystack[i] == needle[j]
```

then the current matched prefix grows by one:

```text
i += 1
j += 1
```

If `j == m`, then all of `needle` has matched.

The match ends at index:

```text
i - 1
```

So it starts at:

```text
i - m
```

Because we scan `haystack` from left to right and never skip a possible earlier
match, this is the first occurrence. Return:

```text
i - m
```

#### Case 2: Mismatch After Some Matched Characters

If:

```text
haystack[i] != needle[j]
```

and:

```text
j > 0
```

then we should not move `i` forward yet.

The current `haystack[i]` has not been matched. It might match a shorter prefix
of `needle` after we fall back.

So we update only `j`:

```text
j = lps[j - 1]
```

This keeps the longest prefix of `needle` that is still compatible with the
characters already seen in `haystack`.

#### Case 3: Mismatch With No Matched Characters

If:

```text
haystack[i] != needle[j]
```

and:

```text
j == 0
```

then no prefix of `needle` is currently matched.

There is nothing to preserve, so move to the next character in `haystack`:

```text
i += 1
```

### 10. Complete Algorithm

1. Let `n = len(haystack)` and `m = len(needle)`.
2. If `m == 0`, return `0`.
3. If `m > n`, return `-1`.
4. Build the `lps` table for `needle`.
5. Scan `haystack` with pointer `i` and `needle` with pointer `j`.
6. On a match, advance both pointers.
7. On a mismatch with `j > 0`, fall back using `lps[j - 1]`.
8. On a mismatch with `j == 0`, advance `i`.
9. If `j == m`, return `i - m`.
10. If the scan ends without a full match, return `-1`.

### 11. Python Code

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n = len(haystack)
        m = len(needle)

        if m == 0:
            return 0

        if m > n:
            return -1

        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if needle[i] == needle[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length > 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

        i = 0
        j = 0

        while i < n:
            if haystack[i] == needle[j]:
                i += 1
                j += 1

                if j == m:
                    return i - m
            elif j > 0:
                j = lps[j - 1]
            else:
                i += 1

        return -1
```

This code does not use Python's built-in substring search. It constructs the
string matching logic explicitly.

### 12. Detailed Example Walkthrough: `"sadbutsad"` and `"sad"`

Input:

```text
haystack = "sadbutsad"
needle   = "sad"
```

First build `lps` for `"sad"`.

```text
needle: s a d
index:  0 1 2
```

No non-empty proper prefix is also a suffix for `"sa"` or `"sad"`, so:

```text
lps = [0, 0, 0]
```

Now scan the haystack.

Initial state:

```text
i = 0
j = 0
```

Compare:

```text
haystack[0] = 's'
needle[0]   = 's'
```

Match:

```text
i = 1
j = 1
```

Compare:

```text
haystack[1] = 'a'
needle[1]   = 'a'
```

Match:

```text
i = 2
j = 2
```

Compare:

```text
haystack[2] = 'd'
needle[2]   = 'd'
```

Match:

```text
i = 3
j = 3
```

Now:

```text
j == len(needle)
```

So a full match has ended just before `i`, at index `2`.

The start is:

```text
i - len(needle) = 3 - 3 = 0
```

Return:

```text
0
```

The later occurrence at index `6` does not matter because the problem asks for
the first occurrence.

### 13. Detailed Example Walkthrough: `"leetcode"` and `"leeto"`

Input:

```text
haystack = "leetcode"
needle   = "leeto"
```

Build `lps` for `"leeto"`.

```text
needle: l e e t o
index:  0 1 2 3 4
lps:    0 0 0 0 0
```

There is no useful prefix-suffix overlap in this `needle`.

Now scan:

```text
haystack: l e e t c o d e
needle:   l e e t o
```

The first four characters match:

```text
haystack[0:4] = "leet"
needle[0:4]   = "leet"
```

Now compare:

```text
haystack[4] = 'c'
needle[4]   = 'o'
```

Mismatch.

Because `j = 4`, fall back:

```text
j = lps[3] = 0
```

Now the current haystack character `'c'` is compared against the beginning of
`needle`.

```text
haystack[4] = 'c'
needle[0]   = 'l'
```

Mismatch at `j = 0`, so advance `i`.

The remaining characters are:

```text
o d e
```

None can start `"leeto"`, so the scan finishes without a full match.

Return:

```text
-1
```

### 14. Why the Algorithm Is Correct

We prove correctness in two parts: the `lps` table is correct, and the search
uses it without skipping the first possible match.

#### LPS Table Correctness

For each index `i`, `lps[i]` should equal the length of the longest proper
prefix of `needle[0:i + 1]` that is also a suffix of that same substring.

The construction maintains this invariant:

```text
Before computing lps[i], length is the best prefix-suffix length that could
possibly be extended by needle[i].
```

If `needle[i] == needle[length]`, then that candidate can be extended by one
character, so assigning:

```text
lps[i] = length + 1
```

is correct.

If the characters do not match, the candidate of size `length` cannot be the
answer for `i`. The next largest candidate must itself be the longest
prefix-suffix of the previous candidate, which is exactly:

```text
lps[length - 1]
```

The algorithm keeps falling back this way until it finds a candidate that can be
extended or reaches `0`. Therefore each `lps[i]` is assigned the longest valid
prefix-suffix length.

#### Search Correctness

During the search, the key invariant is:

```text
haystack[i - j : i] == needle[0:j]
```

That means the last `j` processed characters of `haystack` match the first `j`
characters of `needle`.

If the next characters match, advancing both `i` and `j` preserves the invariant
with a longer matched prefix.

If the next characters do not match and `j > 0`, then a full match ending
through the current position is impossible with the current `j`. But the
already matched characters may still contain a suffix that equals a prefix of
`needle`. The longest such suffix has length:

```text
lps[j - 1]
```

Setting:

```text
j = lps[j - 1]
```

preserves the invariant while discarding only matched characters that cannot be
part of the next candidate.

If the next characters do not match and `j == 0`, no prefix of `needle` is
currently matched. Therefore no match can start at the current `haystack`
position, so advancing `i` is safe.

When `j == m`, the invariant says:

```text
haystack[i - m : i] == needle
```

So `i - m` is a valid occurrence. Since `i` only moves left to right and the
algorithm never discards a candidate start that could still match, this is the
first occurrence.

If the scan ends without reaching `j == m`, no valid start remains. Returning
`-1` is correct.

### 15. Complexity

Let:

```text
n = len(haystack)
m = len(needle)
```

Building the `lps` table takes:

```text
O(m)
```

Although the LPS builder has a loop that sometimes falls back, each fallback
reduces `length`, and each successful match increases it. Across the whole
build, those movements are bounded by `m`.

Scanning the haystack takes:

```text
O(n)
```

The pointer `i` only moves forward. The pointer `j` can move forward on matches
and backward through `lps` on mismatches, but those movements are also bounded
by the amount of previous progress.

Total complexity:

```text
Time:  O(n + m)
Space: O(m)
```

The brute force baseline is:

```text
Time:  O(n * m) in the worst case
Space: O(1)
```

For small inputs, brute force is often accepted and simpler. KMP is the
principled linear-time version that avoids rechecking known matched prefixes.

### 16. Common Pitfalls

- Stopping the outer brute force loop too late or too early. The last possible
  start is `n - m`, so the loop range must include `n - m`.
- Returning the second or later match because the code keeps scanning after the
  first full match. The problem asks for the first occurrence, so return
  immediately.
- Moving `i` forward after every KMP mismatch. When `j > 0`, only `j` should
  fall back; the same `haystack[i]` still needs to be tested against the shorter
  candidate prefix.
- Using `lps[j]` instead of `lps[j - 1]` after a mismatch. If `j` characters
  have matched, the relevant completed prefix ends at index `j - 1`.
- Forgetting the `m > n` case. There cannot be a match if the pattern is longer
  than the text.
- Treating subsequences as substrings. The match must be contiguous.
- Building `lps` with the whole string as its own prefix. `lps[i]` uses a
  proper prefix, so the entire substring is not allowed.
- Assuming KMP is required in every interview. For this problem, a clear
  brute-force implementation may be acceptable, but KMP explains the
  first-principles optimization.

### 17. First-Principles Summary

The problem asks for the leftmost alignment where every character of `needle`
matches the corresponding character of `haystack`.

The brute force solution checks each alignment directly. It is correct because
it tests the exact definition of a substring match from left to right.

The inefficiency is repeated comparison after partial matches. KMP removes that
repetition by remembering how much of the matched prefix can still be useful
after a mismatch.

The core state is:

```text
j = length of the prefix of needle currently matched
```

The core invariant is:

```text
haystack[i - j : i] == needle[0:j]
```

The `lps` table tells the algorithm how far `j` can safely fall back without
forgetting a suffix that may still become the beginning of a valid match.

So the whole algorithm is:

> Scan the text once, grow the matched prefix when characters agree, and on a
> mismatch fall back to the longest prefix that is still consistent with the
> characters already seen.

That is the reason KMP finds the first occurrence in linear time.

## Implementation

See `solutions/array_string/p028_find_the_index_of_the_first_occurrence_in_a_string.py`.

## Tests

See `tests/array_string/test_p028_find_the_index_of_the_first_occurrence_in_a_string.py`.

## Examples

### Example 1
- Input: `{'haystack': 'sadbutsad', 'needle': 'sad'}`
- Output: `0`

### Example 2
- Input: `{'haystack': 'leetcode', 'needle': 'leeto'}`
- Output: `-1`

## Follow-up Practice
- Implement the brute-force scan first and trace every attempted start index.
- Build the `lps` table for `ababaca`, `aaaa`, and `abcab`.
- Trace KMP on a mismatch where `j > 0` and confirm that `i` does not move.
