# 97. Interleaving String

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/interleaving-string/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given three strings:

```text
s1 = first source string
s2 = second source string
s3 = target string
```

The question is whether `s3` can be formed by interleaving `s1` and `s2`.

Interleaving means:

- Every character of `s1` must be used exactly once.
- Every character of `s2` must be used exactly once.
- The relative order of characters inside `s1` must stay the same.
- The relative order of characters inside `s2` must stay the same.
- Characters from `s1` and `s2` may be mixed together in any order as long as those two internal orders are preserved.

For example:

```text
s1 = "abc"
s2 = "def"
```

This is a valid interleaving:

```text
"adbcef"
```

because the characters from `s1` appear as:

```text
a -> b -> c
```

and the characters from `s2` appear as:

```text
d -> e -> f
```

both in their original order.

This is not a valid interleaving:

```text
"abedcf"
```

because the `s2` characters appear as:

```text
e -> d -> f
```

which reverses `d` and `e`.

So the real problem is:

> Can we walk through `s3` from left to right, and for each character decide whether it came from the next unused character of `s1` or the next unused character of `s2`?

---

### 2. The First Necessary Check: Length

If `s3` is an interleaving of `s1` and `s2`, it must use every character from both strings exactly once.

Therefore:

```text
len(s3) must equal len(s1) + len(s2)
```

If the lengths do not match, the answer is immediately `False`.

For example:

```text
s1 = "abc"
s2 = "de"
s3 = "abcdeX"
```

`len(s1) + len(s2) = 5`, but `len(s3) = 6`, so `s3` cannot be formed from only those two strings.

This check does not prove the answer is `True`, but it quickly rejects impossible cases.

---

### 3. Start From the Brute Force Recursion

Imagine building `s3` from left to right.

At any moment, suppose we have already used:

```text
i characters from s1
j characters from s2
```

Then we have built exactly:

```text
i + j characters of s3
```

So the next target character is:

```text
s3[i + j]
```

There are at most two possible moves:

1. If `s1[i] == s3[i + j]`, take the next character from `s1`.
2. If `s2[j] == s3[i + j]`, take the next character from `s2`.

A direct recursive search says:

```python
def can_form(i, j):
    if i == len(s1) and j == len(s2):
        return True

    k = i + j

    if i < len(s1) and s1[i] == s3[k] and can_form(i + 1, j):
        return True

    if j < len(s2) and s2[j] == s3[k] and can_form(i, j + 1):
        return True

    return False
```

This is the cleanest way to understand the problem.

The recursion keeps asking:

> From this pair of positions `(i, j)`, is there any valid way to finish forming the rest of `s3`?

This brute force is correct, but it can repeat the same work many times.

---

### 4. Why Brute Force Repeats Work

The branching happens when both next characters match the next character of `s3`.

For example:

```text
s1 = "aa"
s2 = "aa"
s3 = "aaaa"
```

At the start, the next target character is `a`.

Both choices look valid:

```text
take from s1
take from s2
```

After a few choices, many different paths can reach the same state.

For example, these two histories both lead to the state where one character has been used from each string:

```text
1. take s1, then s2
2. take s2, then s1
```

Both histories arrive at:

```text
i = 1
j = 1
```

At that point, the remaining problem is identical.

It does not matter how we got there. The only relevant facts are:

```text
how many characters of s1 have been consumed
how many characters of s2 have been consumed
```

That is the key sign that this is a dynamic programming problem.

---

### 5. The Key Observation

When we have used `i` characters from `s1` and `j` characters from `s2`, there is no freedom about how many characters of `s3` have been matched.

It must be:

```text
i + j
```

So a state does not need three indices.

We do not need:

```text
(i, j, k)
```

because:

```text
k = i + j
```

This is the central compression:

> A pair `(i, j)` completely determines the prefix of `s3` being considered.

That gives us a two-dimensional DP table.

---

### 6. DP State and Invariant

Define:

```text
dp[i][j] = True if s3[:i + j] can be formed by interleaving s1[:i] and s2[:j]
```

This definition is prefix-based.

It means:

- We use the first `i` characters of `s1`.
- We use the first `j` characters of `s2`.
- Together, they must form the first `i + j` characters of `s3`.
- The internal order of each source prefix must be preserved.

For example:

```text
dp[2][3]
```

means:

```text
Can s1[:2] and s2[:3] interleave to form s3[:5]?
```

The invariant is precise:

> Every `True` cell represents a reachable prefix state; every `False` cell represents a prefix state that cannot be reached while preserving the order of both source strings.

The final answer is:

```text
dp[len(s1)][len(s2)]
```

because that asks whether all of `s1` and all of `s2` can form all of `s3`.

---

### 7. Deriving the Transition From the Last Character

To compute `dp[i][j]`, look at the last character of the target prefix:

```text
s3[:i + j]
```

Its last character is:

```text
s3[i + j - 1]
```

That final character must have come from one of two places:

1. The last used character of `s1`, which is `s1[i - 1]`.
2. The last used character of `s2`, which is `s2[j - 1]`.

#### Case 1: Last Character Comes From `s1`

This is possible if:

```text
i > 0
s1[i - 1] == s3[i + j - 1]
dp[i - 1][j] is True
```

Why `dp[i - 1][j]`?

Because before taking `s1[i - 1]`, we must have already formed:

```text
s3[:i + j - 1]
```

using:

```text
s1[:i - 1]
s2[:j]
```

#### Case 2: Last Character Comes From `s2`

This is possible if:

```text
j > 0
s2[j - 1] == s3[i + j - 1]
dp[i][j - 1] is True
```

Therefore:

```text
dp[i][j] =
    (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1])
    or
    (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
```

with the boundary checks `i > 0` and `j > 0`.

This transition exactly mirrors the brute force choices, but it stores each `(i, j)` result once.

---

### 8. Base Cases

The empty prefixes can always form the empty target prefix:

```text
dp[0][0] = True
```

The first column means using only `s1`:

```text
dp[i][0] = True if s1[:i] == s3[:i]
```

That is because no characters from `s2` are available.

The first row means using only `s2`:

```text
dp[0][j] = True if s2[:j] == s3[:j]
```

That is because no characters from `s1` are available.

The general transition also handles these boundaries if we check array bounds carefully.

---

### 9. Detailed Algorithm

1. Let `m = len(s1)` and `n = len(s2)`.
2. If `m + n != len(s3)`, return `False`.
3. Create a boolean table with `(m + 1)` rows and `(n + 1)` columns.
4. Set `dp[0][0] = True`.
5. For every `i` from `0` to `m`:
   - For every `j` from `0` to `n`:
     - Skip `(0, 0)` because it is already initialized.
     - Let `k = i + j - 1`, the index of the last character in the current `s3` prefix.
     - If `i > 0`, check whether `s1[i - 1]` can be the last character.
     - If `j > 0`, check whether `s2[j - 1]` can be the last character.
     - Mark `dp[i][j]` true if either route works.
6. Return `dp[m][n]`.

The fill order works because `dp[i][j]` depends only on:

```text
dp[i - 1][j]
dp[i][j - 1]
```

Those are the cell above and the cell to the left, which have already been computed when we scan row by row.

---

### 10. Code

A direct Python implementation is:

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m = len(s1)
        n = len(s2)

        if m + n != len(s3):
            return False

        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue

                k = i + j - 1

                from_s1 = (
                    i > 0
                    and dp[i - 1][j]
                    and s1[i - 1] == s3[k]
                )

                from_s2 = (
                    j > 0
                    and dp[i][j - 1]
                    and s2[j - 1] == s3[k]
                )

                dp[i][j] = from_s1 or from_s2

        return dp[m][n]
```

The same idea can also be written recursively with memoization:

```python
from functools import cache

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False

        @cache
        def can_form(i: int, j: int) -> bool:
            if i == len(s1) and j == len(s2):
                return True

            k = i + j

            if i < len(s1) and s1[i] == s3[k] and can_form(i + 1, j):
                return True

            if j < len(s2) and s2[j] == s3[k] and can_form(i, j + 1):
                return True

            return False

        return can_form(0, 0)
```

Both versions solve the same subproblems. The table version builds answers from small prefixes upward; the memoized version explores only needed states and caches them.

---

### 11. Detailed Walkthrough of Example 1

Input:

```text
s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"
```

The lengths match:

```text
len(s1) = 5
len(s2) = 5
len(s3) = 10
```

So we continue.

A successful interleaving path is:

```text
s3:  a a d b b c b c a c
      | | | | | | | | | |
from: 1 1 2 2 1 1 2 2 2 1
```

That means:

```text
s1 contributes: a a b c c
s2 contributes: d b b c a
```

which preserves both original orders.

Now view this through DP states.

Start:

```text
dp[0][0] = True
```

We have formed the empty prefix.

#### Matching the first two `a` characters

The target starts with:

```text
s3[:2] = "aa"
```

Only `s1` starts with two `a` characters. `s2` starts with `d`.

So the reachable states include:

```text
dp[1][0] = True   # "a" from s1 forms "a"
dp[2][0] = True   # "aa" from s1 forms "aa"
```

At this point, we have used:

```text
s1[:2] = "aa"
s2[:0] = ""
```

and formed:

```text
s3[:2] = "aa"
```

#### Taking `d` from `s2`

The next target character is:

```text
s3[2] = "d"
```

The next unused character of `s1` is:

```text
s1[2] = "b"
```

The next unused character of `s2` is:

```text
s2[0] = "d"
```

So the only valid move is to take from `s2`:

```text
dp[2][1] = True
```

Now the matched prefix is:

```text
s3[:3] = "aad"
```

using:

```text
s1[:2] = "aa"
s2[:1] = "d"
```

#### Ambiguity around the `b` characters

The next target characters include several `b` values:

```text
s3[3:6] = "bbc"
```

Both `s1` and `s2` may offer `b` at different moments.

This is exactly where brute force would branch.

The DP table does not commit to only one history. It records every reachable prefix pair.

For example, after forming:

```text
s3[:5] = "aadbb"
```

there can be multiple plausible ways to split the two `b` characters between `s1` and `s2`.

The table keeps all reachable states, such as states that have consumed different counts from `s1` and `s2`, as long as their prefixes can produce the same target prefix.

#### Finishing the valid path

One valid sequence of consumed counts is:

```text
(i, j)
(0, 0)
(1, 0)  take 'a' from s1
(2, 0)  take 'a' from s1
(2, 1)  take 'd' from s2
(2, 2)  take 'b' from s2
(3, 2)  take 'b' from s1
(4, 2)  take 'c' from s1
(4, 3)  take 'b' from s2
(4, 4)  take 'c' from s2
(4, 5)  take 'a' from s2
(5, 5)  take 'c' from s1
```

Since the final state is reachable:

```text
dp[5][5] = True
```

The algorithm returns:

```text
True
```

---

### 12. Why Example 2 Fails

Input:

```text
s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbbaccc"
```

The lengths match, so length alone cannot reject it.

The problem appears near the middle:

```text
s3 = "aadbbbaccc"
              ^
```

There are too many `b` choices before the later `a` and `c` characters can be placed while preserving both source orders.

The DP table explores every prefix split `(i, j)`. Whenever a target character cannot be supplied by the next unused character of either source from a reachable state, that route dies.

Eventually no reachable route reaches:

```text
dp[5][5]
```

So the answer is:

```text
False
```

The important lesson is that matching character counts is not enough. The ordering constraints matter.

---

### 13. Correctness

We prove that the algorithm returns `True` exactly when `s3` is an interleaving of `s1` and `s2`.

#### Lemma 1: Every `True` DP cell represents a valid interleaving of prefixes.

`dp[0][0]` is `True`, and the empty prefixes form the empty string.

For any other cell `dp[i][j]`, the algorithm sets it to `True` only in one of two cases:

- `dp[i - 1][j]` is `True` and `s1[i - 1]` equals the last character of `s3[:i + j]`.
- `dp[i][j - 1]` is `True` and `s2[j - 1]` equals the last character of `s3[:i + j]`.

In the first case, a valid interleaving for `s1[:i - 1]` and `s2[:j]` can be extended by appending `s1[i - 1]`.

In the second case, a valid interleaving for `s1[:i]` and `s2[:j - 1]` can be extended by appending `s2[j - 1]`.

Both extensions preserve the internal order of `s1` and `s2`.

Therefore every `True` cell is valid.

#### Lemma 2: Every valid interleaving of prefixes is marked `True`.

Suppose `s3[:i + j]` is a valid interleaving of `s1[:i]` and `s2[:j]`.

Look at its final character.

That final character must come from either:

- the final character of `s1[:i]`, or
- the final character of `s2[:j]`.

If it comes from `s1[i - 1]`, then the preceding prefix must be a valid interleaving of `s1[:i - 1]` and `s2[:j]`. By induction, `dp[i - 1][j]` is `True`, and the transition sets `dp[i][j]` to `True`.

If it comes from `s2[j - 1]`, then the preceding prefix must be a valid interleaving of `s1[:i]` and `s2[:j - 1]`. By induction, `dp[i][j - 1]` is `True`, and the transition sets `dp[i][j]` to `True`.

Therefore every valid prefix interleaving is represented in the table.

#### Theorem: The algorithm returns the correct answer.

By Lemma 1, if `dp[len(s1)][len(s2)]` is `True`, then all of `s1` and all of `s2` form all of `s3`, so `s3` is a valid interleaving.

By Lemma 2, if `s3` is a valid interleaving, then the final DP cell must be marked `True`.

Thus the returned value is correct.

---

### 14. Complexity

Let:

```text
m = len(s1)
n = len(s2)
```

The DP table has:

```text
(m + 1) * (n + 1)
```

states.

Each state does `O(1)` work.

So the time complexity is:

```text
O(m * n)
```

The table stores `O(m * n)` booleans, so the space complexity is:

```text
O(m * n)
```

The space can be optimized to `O(n)` because each row depends only on the previous row and the current row's left neighbor, but the two-dimensional version is usually the clearest first implementation.

---

### 15. Common Pitfalls

#### Forgetting the Length Check

Without this check:

```python
if len(s1) + len(s2) != len(s3):
    return False
```

indexing logic can become awkward, and impossible inputs may be processed unnecessarily.

#### Treating Interleaving as Just Character Counts

The strings:

```text
s1 = "ab"
s2 = "cd"
s3 = "badc"
```

have the same combined characters, but `s3` is not valid because `b` appears before `a`, breaking the order of `s1`.

Order matters.

#### Using One Index Instead of Two

A single position in `s3` is not enough to know the state.

At target index `k`, we also need to know how many of those `k` characters came from `s1` and how many came from `s2`.

That is why the state must be `(i, j)`.

#### Off-by-One Errors

For prefix DP:

```text
dp[i][j] talks about prefixes of length i and j
```

Therefore the last used characters are:

```text
s1[i - 1]
s2[j - 1]
s3[i + j - 1]
```

Only access those when the corresponding length is positive.

#### Thinking a Greedy Choice Is Enough

When both `s1[i]` and `s2[j]` match the next character of `s3`, choosing one greedily can block a valid solution later.

The DP table keeps both possibilities alive by marking all reachable states.

---

### 16. First-Principles Summary

The problem is about preserving two independent orders while producing one target order.

A partial construction is fully described by:

```text
how many characters were consumed from s1
how many characters were consumed from s2
```

Those two counts determine the target prefix length, because:

```text
matched characters in s3 = i + j
```

From state `(i, j)`, the next or last character can only come from the next or last unused character of `s1` or `s2`.

That gives the whole recurrence:

```text
reachable from above if the last character came from s1
reachable from left if the last character came from s2
```

The DP table is not a generic trick here. It is a compact record of all possible ways to split each prefix of `s3` between the prefixes of `s1` and `s2` without ever violating source order.

## Implementation

See `solutions/dynamic_programming_multidimensional/p097_interleaving_string.py`.

## Tests

See `tests/dynamic_programming_multidimensional/test_p097_interleaving_string.py`.

## Examples

### Example 1
- Input: `{'s1': 'aabcc', 's2': 'dbbca', 's3': 'aadbbcbcac'}`
- Output: `True`

### Example 2
- Input: `{'s1': 'aabcc', 's2': 'dbbca', 's3': 'aadbbbaccc'}`
- Output: `False`

### Example 3
- Input: `{'s1': '', 's2': '', 's3': ''}`
- Output: `True`

## Follow-up Practice

- Trace the recursion tree for `s1 = "aa"`, `s2 = "aa"`, and `s3 = "aaaa"`, then identify repeated states.
- Draw the DP table for `s1 = "ab"`, `s2 = "cd"`, and `s3 = "acbd"`.
- Explain why `dp[i][j]` uses `s3[i + j - 1]` instead of `s3[i + j]` in the table transition.
- Rewrite the table solution using a one-dimensional rolling array after the two-dimensional invariant feels natural.
