# 139. Word Break

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/word-break/
- Official Group: 1D DP
- Pattern Group: Dynamic Programming 1D
- Patterns: dynamic-programming-1d

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given:

```text
s        = a string
wordDict = a list of allowed words
```

Return whether `s` can be split into one or more dictionary words.

You may reuse dictionary words as many times as needed.

For example:

```text
s = "leetcode"
wordDict = ["leet", "code"]
```

The string can be split as:

```text
"leet" + "code"
```

Both pieces are in the dictionary, so the answer is `True`.

For:

```text
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
```

Some prefixes look promising:

```text
"cats" + "and" + ...
"cat" + "sand" + ...
```

But both paths leave the suffix:

```text
"og"
```

which cannot be formed from the dictionary. So the answer is `False`.

The real problem is:

> Is there at least one sequence of dictionary words whose concatenation is exactly `s`?

This is not asking for the actual split. It only asks whether such a split exists.

---

### 2. Start From the Brute Force Recursion

The most literal way to solve the problem is to try every possible first word.

Suppose:

```text
s = "applepenapple"
wordDict = ["apple", "pen"]
```

At the beginning, ask:

```text
Can the whole string be broken?
```

That becomes:

```text
Is there a word in wordDict that matches the front of s?
```

`"apple"` matches the front, so the question reduces to:

```text
Can "penapple" be broken?
```

Then `"pen"` matches the front, so the question reduces to:

```text
Can "apple" be broken?
```

Then `"apple"` matches, leaving the empty suffix:

```text
Can "" be broken?
```

The empty suffix means every previous character has been successfully covered, so that path succeeds.

Conceptually:

```python
def can_break(start):
    if start == len(s):
        return True

    for word in wordDict:
        if s.startswith(word, start):
            if can_break(start + len(word)):
                return True

    return False
```

This is correct because it tries every possible first word, then every possible next word, and so on.

But it can be very slow.

The same suffix can be reached by multiple split paths.

For example, with words such as:

```text
["a", "aa", "aaa", "aaaa"]
```

and a string made of many `a` characters, the recursion repeatedly asks questions like:

```text
Can the suffix starting at index 7 be broken?
Can the suffix starting at index 8 be broken?
Can the suffix starting at index 9 be broken?
```

The answer to `can_break(7)` does not depend on how we reached index `7`. It depends only on the remaining suffix `s[7:]`.

That repeated work is the signal that dynamic programming applies.

---

### 3. The Key Observation

Every valid split has a final word.

If `s[0:i]` can be broken into dictionary words, then its last word must be some substring:

```text
s[j:i]
```

for some `j < i`.

Then two things must both be true:

```text
1. s[0:j] can already be broken into dictionary words.
2. s[j:i] is itself a dictionary word.
```

So instead of thinking forward:

```text
Choose the next word from the current position.
```

we can think backward for each prefix:

```text
Where could the last word of this prefix start?
```

That turns the problem into a sequence of prefix questions.

If we know which earlier prefixes are breakable, we can decide whether the next prefix is breakable.

---

### 4. DP State and Invariant

Define:

```text
dp[i] = True if and only if s[0:i] can be segmented into dictionary words
```

Important details:

- `i` is a length, not a character index.
- `s[0:i]` means the prefix ending before index `i`.
- `dp[len(s)]` answers the original problem.

The base case is:

```text
dp[0] = True
```

Why is the empty prefix breakable?

Because it represents the state before using any words. It is the neutral starting point that allows a dictionary word at the beginning of `s` to be accepted.

For example, if:

```text
s[0:4] = "leet"
```

then `dp[4]` should become `True` because:

```text
dp[0] is True
s[0:4] is "leet", a dictionary word
```

The invariant we maintain is:

> After computing `dp[i]`, it exactly records whether the prefix `s[0:i]` can be formed by concatenating dictionary words.

No extra information is needed. We do not need to remember the exact words used, because the question only asks for a boolean answer.

---

### 5. Transition: Try the Last Cut

To compute `dp[i]`, try every possible previous cut position `j`:

```text
s[0:i] = s[0:j] + s[j:i]
```

The prefix `s[0:i]` is breakable if there exists at least one `j` such that:

```text
dp[j] == True
and
s[j:i] is in wordDict
```

So the transition is:

```text
dp[i] = any(dp[j] and s[j:i] in wordSet for j in range(i))
```

This directly matches the last-word observation.

The dictionary should be stored in a set so membership checks are fast:

```python
word_set = set(wordDict)
```

Without a set, checking whether a substring is in `wordDict` would require scanning the list each time.

---

### 6. Detailed Algorithm

1. Let `n = len(s)`.
2. Convert `wordDict` to a set called `word_set`.
3. Create a boolean array `dp` of length `n + 1`, initialized to `False`.
4. Set `dp[0] = True`.
5. For each prefix length `i` from `1` to `n`:
   - Try every cut position `j` from `0` to `i - 1`.
   - If `dp[j]` is `False`, then `s[0:j]` cannot be built, so this cut cannot help.
   - If `dp[j]` is `True`, check whether `s[j:i]` is a dictionary word.
   - If both are true, set `dp[i] = True` and stop checking more cuts for this `i`.
6. Return `dp[n]`.

The early stop is safe because the problem only asks whether at least one valid segmentation exists.

---

### 7. Pseudocode

```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)

    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

A common small optimization is to avoid checking impossible word lengths.

If the longest dictionary word has length `max_len`, then for a fixed `i`, there is no need to consider `j < i - max_len`.

That optimized version is:

```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    max_len = max((len(word) for word in wordDict), default=0)
    n = len(s)

    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        start = max(0, i - max_len)
        for j in range(start, i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

Both versions use the same invariant. The second version only skips substrings that are longer than every dictionary word, so they cannot possibly match.

---

### 8. Walk Through Example 1

Input:

```text
s = "leetcode"
wordDict = ["leet", "code"]
```

The string has length `8`, so `dp` has indices `0` through `8`.

Start:

```text
dp[0] = True
dp[1..8] = False
```

Now compute each prefix.

#### Prefix length 1: `s[0:1] = "l"`

Possible last word:

```text
"l"
```

It is not in the dictionary, so:

```text
dp[1] = False
```

#### Prefix length 2: `s[0:2] = "le"`

Try cuts:

```text
"le"
"e"
```

No valid dictionary word completes a breakable prefix, so:

```text
dp[2] = False
```

#### Prefix length 3: `s[0:3] = "lee"`

No valid cut works:

```text
dp[3] = False
```

#### Prefix length 4: `s[0:4] = "leet"`

Try `j = 0`:

```text
dp[0] = True
s[0:4] = "leet"
```

`"leet"` is in the dictionary, so:

```text
dp[4] = True
```

This means the prefix `"leet"` can be broken.

#### Prefix lengths 5, 6, and 7

These prefixes are:

```text
"leetc"
"leetco"
"leetcod"
```

Even though `dp[4]` is `True`, the suffixes after index `4` are:

```text
"c"
"co"
"cod"
```

None of those is a dictionary word, so:

```text
dp[5] = False
dp[6] = False
dp[7] = False
```

#### Prefix length 8: `s[0:8] = "leetcode"`

Try cut `j = 4`:

```text
dp[4] = True
s[4:8] = "code"
```

`"code"` is in the dictionary, so:

```text
dp[8] = True
```

The final answer is:

```text
dp[len(s)] = dp[8] = True
```

The successful split is:

```text
"leet" + "code"
```

---

### 9. Walk Through Example 3

Input:

```text
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
```

Useful breakable prefixes are found early:

```text
dp[3] = True   because "cat" is a word
dp[4] = True   because "cats" is a word
dp[7] = True   because "cat" + "sand" works
               or "cats" + "and" works
```

So far, the string prefix:

```text
"catsand"
```

can be segmented.

But the full string is:

```text
"catsandog"
```

To make `dp[9]` true, there must be a final dictionary word ending at index `9`.

The natural candidate is:

```text
s[6:9] = "dog"
```

`"dog"` is a dictionary word, but it would require:

```text
dp[6] = True
```

That would mean the prefix `s[0:6]`, which is `"catsan"`, can be segmented. It cannot.

Another possible suffix is:

```text
"og"
```

but `"og"` is not a dictionary word.

Every possible final cut fails, so:

```text
dp[9] = False
```

The answer is `False`.

This example shows why a greedy approach is unsafe. Choosing `"cats"` or choosing `"cat"` can both look reasonable, but local choices do not prove the whole string is breakable. The DP avoids committing to one path too early.

---

### 10. Why Greedy Does Not Work

A tempting idea is:

```text
Always take the longest dictionary word that matches the current prefix.
```

But this can fail.

Example:

```text
s = "cars"
wordDict = ["car", "ca", "rs"]
```

The longest matching first word is:

```text
"car"
```

That leaves:

```text
"s"
```

which is not a word.

But the correct split is:

```text
"ca" + "rs"
```

So the algorithm must keep enough information to recover from locally attractive choices. The boolean DP does that by considering every possible last cut for each prefix.

---

### 11. Correctness

We prove that the algorithm returns `True` if and only if `s` can be segmented into dictionary words.

#### Lemma 1: If `dp[i]` is `True`, then `s[0:i]` can be segmented into dictionary words.

The only way the algorithm sets `dp[i]` to `True` is by finding some cut `j` such that:

```text
dp[j] is True
s[j:i] is in word_set
```

By the meaning of `dp[j]`, the prefix `s[0:j]` can be segmented into dictionary words. The substring `s[j:i]` is also a dictionary word. Concatenating the segmentation of `s[0:j]` with `s[j:i]` gives a valid segmentation of `s[0:i]`.

Therefore, whenever `dp[i]` is `True`, the prefix `s[0:i]` is truly breakable.

#### Lemma 2: If `s[0:i]` can be segmented into dictionary words, then `dp[i]` becomes `True`.

Assume `s[0:i]` has a valid segmentation. Look at the last word in that segmentation.

Suppose the last word starts at index `j`, so it is:

```text
s[j:i]
```

Then:

```text
s[0:j]
```

is also validly segmented, because it is everything before the last word.

Since `j < i`, the algorithm computes `dp[j]` before `dp[i]`. By the invariant, `dp[j]` is `True`. Also, the last word `s[j:i]` is in the dictionary.

When the algorithm checks this cut `j`, it sets `dp[i]` to `True`.

Therefore, every breakable prefix is marked `True`.

#### Conclusion

By Lemma 1 and Lemma 2, for every `i`, `dp[i]` is `True` exactly when `s[0:i]` can be segmented into dictionary words.

The original question asks whether the entire string `s[0:len(s)]` can be segmented, so returning `dp[len(s)]` is correct.

---

### 12. Complexity

Let:

```text
n = len(s)
m = number of words in wordDict
L = maximum word length
```

The basic DP considers every pair `(j, i)` with `0 <= j < i <= n`, so there are `O(n^2)` cuts.

With a hash set, dictionary membership is average `O(1)` after the substring is created.

In Python, slicing `s[j:i]` creates a new string of length `i - j`, so the strict worst-case time can include substring-copying cost. In common LeetCode-style analysis, this solution is usually described as:

```text
Time:  O(n^2) checks, with substring costs depending on language/runtime
Space: O(n + m) for the dp array and word set
```

With the `max_len` optimization, the inner loop checks at most `L` cuts for each `i`, so it becomes:

```text
Time:  O(n * L) membership checks, again with substring-copying cost
Space: O(n + m)
```

The `dp` array itself uses `O(n)` space.

---

### 13. Common Pitfalls

- Forgetting `dp[0] = True`. Without it, words that start at index `0` can never make any prefix valid.
- Defining `dp[i]` as a character position but then slicing with prefix-length semantics. Keep `dp[i]` tied to `s[0:i]`.
- Returning too early on a failed cut. A prefix is false only after every possible final cut has failed.
- Using greedy longest-prefix matching. A locally longest word can block a valid later split.
- Checking `s[j:i] in wordDict` when `wordDict` is a list. Convert it to a set for efficient membership checks.
- Thinking dictionary words can be used only once. In this problem, words may be reused.
- Confusing this problem with counting segmentations. Here `dp[i]` is boolean because only existence matters.
- Missing the empty string edge case. If `s` is empty, `dp[0]` correctly represents that no words are needed.

---

### 14. First-Principles Summary

The string can be segmented if some final dictionary word finishes it.

For any prefix `s[0:i]`, the final word must be `s[j:i]` for some earlier cut `j`. That cut is valid only if the earlier prefix `s[0:j]` was already segmentable.

So the whole problem reduces to one repeated question:

```text
Can this prefix be formed from earlier breakable prefixes plus one dictionary word?
```

The DP array records exactly those prefix answers:

```text
dp[i] = whether s[0:i] is breakable
```

Once this invariant is clear, the implementation is just a direct translation:

```text
try every end index i
try every last cut j
accept if dp[j] is true and s[j:i] is a word
```

That is the essence of Word Break.

## Implementation
See `solutions/dynamic_programming_1d/p139_word_break.py`.

## Tests
See `tests/dynamic_programming_1d/test_p139_word_break.py`.

## Examples

### Example 1
- Input: `{'s': 'leetcode', 'wordDict': ['leet', 'code']}`
- Output: `True`

### Example 2
- Input: `{'s': 'applepenapple', 'wordDict': ['apple', 'pen']}`
- Output: `True`

### Example 3
- Input: `{'s': 'catsandog', 'wordDict': ['cats', 'dog', 'sand', 'and', 'cat']}`
- Output: `False`

## Follow-up Practice
- Modify the boolean DP to count how many valid segmentations exist.
- Modify the DP to reconstruct one valid segmentation, not just return `True` or `False`.
- Compare top-down memoization on `start` with bottom-up tabulation on prefix length.
