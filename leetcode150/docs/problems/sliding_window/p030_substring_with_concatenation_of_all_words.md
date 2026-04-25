# 30. Substring with Concatenation of All Words

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/substring-with-concatenation-of-all-words/
- Official Group: Sliding Window
- Pattern Group: Sliding Window
- Patterns: sliding-window, window-or-prefix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given:

```text
s = a string
words = a list of words, all with the same length
```

Find every starting index `i` such that the substring beginning at `i` is made by concatenating every word in `words` exactly once, in any order.

For example:

```text
s = "barfoothefoobarman"
words = ["foo", "bar"]
```

Each word has length `3`, and there are `2` words, so any valid substring must have total length:

```text
3 * 2 = 6
```

At index `0`:

```text
s[0:6] = "barfoo"
```

Split into word-sized chunks:

```text
"bar" + "foo"
```

That uses every word exactly once, so `0` is valid.

At index `9`:

```text
s[9:15] = "foobar"
```

Split into chunks:

```text
"foo" + "bar"
```

That is also valid, so `9` is valid.

The answer is:

```text
[0, 9]
```

The order of words does not matter, but the count of each word does matter.

So the real problem is:

> Find every contiguous block of length `len(words) * len(words[0])` whose word-sized chunks form the same multiset as `words`.

---

### 2. Why This Is Not Just a Character Problem

A normal substring problem often moves one character at a time.

But here, the smallest meaningful unit is not a character. It is a word-sized chunk.

If:

```text
word_len = 3
```

then a valid candidate must be split like this:

```text
[0:3], [3:6], [6:9], ...
```

or like this:

```text
[1:4], [4:7], [7:10], ...
```

or like this:

```text
[2:5], [5:8], [8:11], ...
```

The split depends on the starting index modulo `word_len`.

That is the first key observation:

> Because all words have equal length, a valid substring is a sequence of fixed-size tokens.

So we should reason in tokens, not individual characters.

---

### 3. The Brute Force Baseline

The most direct approach is:

1. Try every starting index `i`.
2. Take the substring of total required length.
3. Split it into word-sized chunks.
4. Count those chunks.
5. Compare the chunk counts with `words` counts.

Conceptually:

```python
need = Counter(words)
total_len = len(words) * len(words[0])
word_len = len(words[0])

for i in range(len(s) - total_len + 1):
    chunk_counts = Counter()

    for j in range(i, i + total_len, word_len):
        chunk = s[j:j + word_len]
        chunk_counts[chunk] += 1

    if chunk_counts == need:
        answer.append(i)
```

This is correct, but it repeats a lot of work.

Adjacent candidates overlap heavily. For example, if `word_len = 3`, the candidate starting at `0` and the candidate starting at `3` share most of their word-sized chunks.

So the deeper question is:

> Can we reuse the token counts from one candidate window when moving to the next candidate window?

Yes. That is exactly what sliding window does.

---

### 4. Why We Need Multiple Sliding Windows

Suppose:

```text
word_len = 3
```

A valid answer can start at any index:

```text
0, 1, 2, 3, 4, 5, ...
```

But once a starting index is chosen, every following token boundary is forced.

If the start is `0`, chunks are:

```text
s[0:3], s[3:6], s[6:9], ...
```

If the start is `1`, chunks are:

```text
s[1:4], s[4:7], s[7:10], ...
```

If the start is `2`, chunks are:

```text
s[2:5], s[5:8], s[8:11], ...
```

These are different alignment classes.

There are exactly `word_len` possible alignments:

```text
offset = 0
offset = 1
...
offset = word_len - 1
```

So instead of one character-based sliding window, we run `word_len` token-based sliding windows.

This is the second key observation:

> Every valid start belongs to exactly one alignment class modulo `word_len`, so checking all offsets covers all possible answers.

---

### 5. What Makes a Token Window Valid?

Let:

```text
need = counts of words in words
seen = counts of words inside the current window
```

A window is valid when:

```text
seen == need
```

Equivalently:

```text
1. The window contains exactly len(words) tokens.
2. Every token is one of the required words.
3. No token appears more times than required.
```

For example:

```text
words = ["foo", "bar", "foo"]
```

Then:

```text
need = {
  "foo": 2,
  "bar": 1
}
```

A valid token window could be:

```text
"bar", "foo", "foo"
```

But this is not valid:

```text
"bar", "foo", "bar"
```

because `"bar"` appears too many times and `"foo"` appears too few times.

So this problem is a multiset-matching problem, not a set-matching problem.

---

### 6. The Window Invariant

For each alignment, we maintain a token window:

```text
s[left:right]
```

where both `left` and `right` move in steps of `word_len`.

The invariant is:

```text
seen records the exact word counts in the current token window,
and no word count in seen exceeds its required count in need.
```

This invariant is powerful because:

- If an invalid word appears, no valid answer can include it.
- If a valid word appears too many times, the window can be repaired by removing tokens from the left.
- If the window contains exactly `len(words)` tokens while respecting all counts, it must be a valid answer.

---

### 7. How the Window Moves

For one alignment:

1. Read the next token at `right`.
2. If the token is not in `need`, reset the window.
3. If the token is in `need`, add it to `seen`.
4. If this token now appears too many times, move `left` forward by whole tokens until the excess is removed.
5. If the window has exactly `len(words)` tokens, record `left` as a valid start.
6. Remove the leftmost token and move `left` forward once so overlapping matches can still be found.

The important point is that the window always moves by whole words, not by single characters inside an alignment.

---

### 8. Example: `barfoothefoobarman`

Let:

```text
s = "barfoothefoobarman"
words = ["foo", "bar"]
word_len = 3
word_count = 2
need = {"foo": 1, "bar": 1}
```

We check offsets `0`, `1`, and `2`.

#### Offset 0

Split into 3-character tokens:

```text
bar | foo | the | foo | bar | man
```

Start with an empty window.

Read `bar`:

```text
seen = {"bar": 1}
window = ["bar"]
```

Read `foo`:

```text
seen = {"bar": 1, "foo": 1}
window = ["bar", "foo"]
```

The window has `2` tokens, and the counts equal `need`, so index `0` is valid.

Record:

```text
answer = [0]
```

Then remove the leftmost token so we can search for overlapping answers:

```text
window = ["foo"]
seen = {"foo": 1}
left = 3
```

Read `the`:

```text
"the" is not in need
```

No valid answer can cross this token, so reset:

```text
window = []
seen = {}
left = 9
```

Read `foo`:

```text
window = ["foo"]
seen = {"foo": 1}
```

Read `bar`:

```text
window = ["foo", "bar"]
seen = {"foo": 1, "bar": 1}
```

Counts match `need`, so index `9` is valid.

Record:

```text
answer = [0, 9]
```

#### Offsets 1 and 2

Their token splits do not produce the required multiset, so they record nothing.

Final answer:

```text
[0, 9]
```

---

### 9. Example With Duplicate Words

Duplicate words are where many incorrect solutions fail.

Let:

```text
s = "wordgoodgoodgoodbestword"
words = ["word", "good", "best", "good"]
```

Then:

```text
need = {
  "word": 1,
  "good": 2,
  "best": 1
}
```

A valid substring must contain two copies of `"good"`, not just one.

The valid answer starts at index `8`:

```text
good | good | best | word
```

The counts are:

```text
{
  "good": 2,
  "best": 1,
  "word": 1
}
```

That equals `need`, so index `8` is valid.

This shows why `words` must be treated as a multiset.

---

### 10. Code

```python
from collections import Counter
from typing import List


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []

        word_len = len(words[0])
        word_count = len(words)
        total_len = word_len * word_count

        if len(s) < total_len:
            return []

        need = Counter(words)
        answer = []

        for offset in range(word_len):
            left = offset
            seen = Counter()
            used = 0

            for right in range(offset, len(s) - word_len + 1, word_len):
                word = s[right:right + word_len]

                if word not in need:
                    seen.clear()
                    used = 0
                    left = right + word_len
                    continue

                seen[word] += 1
                used += 1

                while seen[word] > need[word]:
                    left_word = s[left:left + word_len]
                    seen[left_word] -= 1
                    used -= 1
                    left += word_len

                if used == word_count:
                    answer.append(left)

                    left_word = s[left:left + word_len]
                    seen[left_word] -= 1
                    used -= 1
                    left += word_len

        return answer
```

---

### 11. Why This Code Is Correct

The proof follows from the window invariant.

For each offset, `left` and `right` always stay on valid token boundaries for that alignment. Therefore every window considered by that pass is a sequence of word-sized chunks.

The `seen` counter exactly represents the words inside the current token window.

If the algorithm sees a token that is not in `need`, then no valid concatenation can include that token. Because every candidate in this alignment is made of whole tokens, any candidate crossing that invalid token is impossible. Resetting the window after it is safe.

If the algorithm sees a required token too many times, then any valid window ending at the current `right` must remove tokens from the left until the extra copy is gone. The algorithm does exactly that with the `while seen[word] > need[word]` loop.

After this repair, every count in `seen` is at most its required count.

When `used == word_count`, the window contains exactly the required number of tokens. Since no token count exceeds its required count, and the total number of tokens equals the total required number of tokens, the counts must equal `need`. Therefore `left` is a valid starting index.

Conversely, every valid answer belongs to one offset class. When that offset is processed, the right boundary eventually reaches the end of that valid block. Since the block contains only required words and no overrepresented word, the window will not be reset or shrunk past its valid start before being recorded. Therefore every valid answer is found.

So the algorithm returns exactly all valid starting indices.

---

### 12. Why It Is Efficient

For each offset:

```text
right moves forward by word_len each time
left also only moves forward by word_len
```

No token boundary is processed many times in the same offset.

Across all offsets, the number of token reads is proportional to `len(s)`.

Complexity:

```text
Time:  O(n * w) in Python if slicing a word of length w costs O(w)
       Often described as O(n) when word extraction is treated as bounded or constant

Space: O(m)
```

where:

```text
n = len(s)
m = len(words)
w = len(words[0])
```

The extra space is for the counters storing required and currently seen words.

---

### 13. Common Pitfalls

#### Pitfall 1: Treating `words` as a set

This fails when there are duplicate words.

```text
words = ["good", "good", "best", "word"]
```

You need counts, not just membership.

#### Pitfall 2: Checking only offset `0`

Valid answers can start at any index modulo `word_len`.

If `word_len = 3`, you must check offsets:

```text
0, 1, 2
```

#### Pitfall 3: Moving by one character inside a token window

Inside a fixed alignment, the natural unit is one word.

So movement should be:

```text
left += word_len
right += word_len
```

#### Pitfall 4: Resetting incorrectly after an invalid token

If a token is not in `need`, no valid window can cross it.

So the correct reset is:

```text
seen = empty
used = 0
left = right + word_len
```

#### Pitfall 5: Forgetting overlapping answers

After recording a valid start, remove exactly one leftmost token and continue.

Otherwise, overlapping valid substrings may be missed.

---

### 14. First-Principles Summary

This problem follows from these basic facts:

```text
1. Every word has the same length.
2. Therefore, every valid substring can be split into equal-sized tokens.
3. A valid substring has exactly len(words) tokens.
4. Those tokens must match the multiset of words.
5. Token boundaries depend on the starting index modulo word_len.
6. So we run one token-based sliding window per offset.
7. Each window maintains token counts and repairs itself when a word is missing, invalid, or overrepresented.
```

In one sentence:

> Convert the string into word-sized token streams by alignment, then use a sliding window over tokens to find every window whose token multiset equals the multiset of `words`.

## Implementation

See `solutions/sliding_window/p030_substring_with_concatenation_of_all_words.py`.

## Tests

See `tests/sliding_window/test_p030_substring_with_concatenation_of_all_words.py`.

## Examples

### Example 1
- Input: `{'s': 'barfoothefoobarman', 'words': ['foo', 'bar']}`
- Output: `[0, 9]`

### Example 2
- Input: `{'s': 'wordgoodgoodgoodbestword', 'words': ['word', 'good', 'best', 'word']}`
- Output: `[]`

### Example 3
- Input: `{'s': 'barfoofoobarthefoobarman', 'words': ['bar', 'foo', 'the']}`
- Output: `[6, 9, 12]`

## Follow-up Practice
- Trace each offset separately for `word_len = 3`.
- Test duplicate words to confirm the multiset logic.
- Compare the token-window solution with brute force at every starting index.
