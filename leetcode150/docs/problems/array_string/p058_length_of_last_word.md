# 58. Length of Last Word

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/length-of-last-word/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: string-scanning

## First-Principles Explanation

### 1. What the Problem Asks

You are given a string `s` made of words and spaces. A word is any maximal block of non-space characters. The task is to return the length of the **last** word.

Two details matter immediately:

- Words are separated by spaces.
- The string may end with extra spaces, and those spaces do **not** count as a word.

So for:

```text
" fly me to the moon "
```

the last visible characters are spaces, but the last word is still `"moon"`, whose length is `4`.

This means the problem is really:

> Ignore trailing spaces, then measure the contiguous block of non-space characters just before them.

### 2. Start With a Baseline Idea

The most direct way to think about the string is:

1. Break it into words.
2. Take the last word.
3. Return its length.

In Python, that idea often becomes something like:

```python
words = s.split()
return len(words[-1])
```

That works because `split()` discards repeated spaces and produces only the words.

Why is this only a baseline and not the best first-principles solution?

- It builds a whole list of every word even though we need only the last one.
- It does more allocation than necessary.
- It hides the actual structure of the problem: a suffix question about the end of the string.

The input already gives us the answer near the end, so we should read the string in the direction where the answer becomes obvious fastest.

### 3. Key Observation

The answer depends only on the suffix of the string:

- trailing spaces at the end
- then one contiguous run of non-space characters

Everything before that last run is irrelevant once we have found where the last word begins.

That gives the main insight:

> Scanning from right to left lets us ignore useless trailing spaces first, then count exactly the characters that belong to the last word.

We never need to identify earlier words at all.

### 4. Invariant and State

We only need one index and one counter.

State:

- `i`: current index while scanning from right to left
- `length`: number of characters counted in the last word so far

Invariant during the algorithm:

1. After the first phase, every position to the right of `i` is a trailing space and can never belong to the last word.
2. During the counting phase, every character counted into `length` is part of the last word.
3. As soon as we stop counting because we hit a space or the beginning of the string, we have counted the entire last word and nothing else.

This invariant is simple but powerful. It works because once trailing spaces are removed, the first non-space character we meet from the right must belong to the last word, and every consecutive non-space character to its left also belongs to that same word.

### 5. Detailed Algorithm

Use two phases on the same right-to-left scan.

#### Phase 1: Skip trailing spaces

Start from the last index:

```text
i = len(s) - 1
```

While `i >= 0` and `s[i] == ' '`, move left.

When this phase ends:

- either `i < 0` and the string had no word at all
- or `s[i]` is the last character of the last word

#### Phase 2: Count the last word

Initialize:

```text
length = 0
```

While `i >= 0` and `s[i] != ' '`, do:

- increment `length`
- move `i` left

When this phase stops, we have reached:

- a space just before the last word, or
- the beginning of the string

In both cases, `length` is exactly the answer.

### 6. Pseudocode

```python
def length_of_last_word(s):
    i = len(s) - 1

    while i >= 0 and s[i] == ' ':
        i -= 1

    length = 0
    while i >= 0 and s[i] != ' ':
        length += 1
        i -= 1

    return length
```

### 7. Why This Is Better Than Left-to-Right Counting

You could scan left to right and keep resetting a running word length whenever you see spaces. That also works.

But the right-to-left version matches the problem statement more directly:

- we only care about the last word
- trailing spaces are the main annoyance
- starting from the end eliminates irrelevant prefix information immediately

This is a good first-principles habit:

> If the answer is defined by a suffix, consider scanning from the end.

### 8. Detailed Walkthrough

#### Example 1: `s = "Hello World"`

Index the characters:

```text
H e l l o _ W o r l d
0 1 2 3 4 5 6 7 8 9 10
```

Here `_` represents a space.

Start:

```text
i = 10
s[i] = 'd'
```

Phase 1, skip trailing spaces:

- `s[10]` is not a space, so we skip nothing

Phase 2, count the last word:

- `d` -> `length = 1`
- `l` -> `length = 2`
- `r` -> `length = 3`
- `o` -> `length = 4`
- `W` -> `length = 5`

Now `i` moves to index `5`, which is a space, so we stop.

Answer:

```text
5
```

#### Example 2: `s = " fly me to the moon "`

Write the interesting suffix:

```text
... _ m o o n _
            ^
            start from the end
```

Phase 1, skip trailing spaces:

- last character is a space, so move left once
- now `i` points to `'n'`

Phase 2, count:

- `n` -> `length = 1`
- `o` -> `length = 2`
- `o` -> `length = 3`
- `m` -> `length = 4`

The next character to the left is a space, so the last word is complete.

Answer:

```text
4
```

#### Example 3: `s = "luffy is still joyboy"`

There are no trailing spaces, so phase 1 does nothing.

Count backward:

- `y` -> `1`
- `o` -> `2`
- `b` -> `3`
- `y` -> `4`
- `o` -> `5`
- `j` -> `6`

The next character is a space, so the count stops.

Answer:

```text
6
```

### 9. Correctness

We can justify the algorithm in two short steps.

#### Lemma 1

After phase 1 finishes, if `i >= 0`, then `s[i]` is the last character of the last word.

Why?

- Phase 1 moves left past every trailing space.
- It stops at the first non-space character from the right.
- Any non-space character after that would contradict the fact that we already skipped all trailing spaces.

So the first non-space character found from the right must belong to the last word, and specifically must be its last character.

#### Lemma 2

Phase 2 counts exactly all characters of the last word.

Why?

- It starts at the last character of the last word by Lemma 1.
- It keeps moving left while characters are non-space, so every counted character is in the same maximal block of non-space characters.
- It stops only when it reaches a space or the start of the string, which is exactly where that word ends on the left.

Therefore, the final `length` is exactly the length of the last word.

### 10. Complexity

- Time: `O(n)` in the worst case, where `n` is the length of `s`
- Space: `O(1)` extra space

Even though we scan from the end, each character is examined at most once in a meaningful way.

### 11. Common Pitfalls

#### Forgetting trailing spaces

If you begin counting immediately from the last index, inputs like:

```text
"a "
```

will give the wrong answer unless you skip the space first.

#### Stopping after the first space from the end

The first space from the end is not the answer boundary if there are trailing spaces. You must remove those first, then count the word.

#### Using `split(" ")` instead of `split()`

In Python, `split(" ")` keeps empty strings created by repeated spaces, which is awkward. `split()` handles arbitrary whitespace more cleanly. But for this problem, the right-to-left scan is simpler and uses constant extra space.

#### Off-by-one mistakes

Be careful with:

- starting at `len(s) - 1`, not `len(s)`
- checking `i >= 0` before indexing
- incrementing the count only for non-space characters in the final word

### 12. Reference Implementation

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        i = len(s) - 1

        while i >= 0 and s[i] == " ":
            i -= 1

        length = 0
        while i >= 0 and s[i] != " ":
            length += 1
            i -= 1

        return length
```

### 13. First-Principles Summary

This problem looks trivial once you see its shape:

- the answer lives at the end of the string
- trailing spaces are noise
- the last word is one contiguous block of non-space characters

So the cleanest reasoning is:

1. Walk left past the noise.
2. Count the block that remains.
3. Stop at the first separator.

That is the whole algorithm. No list of words, no extra storage, and no need to process earlier words that cannot affect the answer.

## Implementation

See `solutions/array_string/p058_length_of_last_word.py`.

## Tests

See `tests/array_string/test_p058_length_of_last_word.py`.

## Examples

### Example 1
- Input: `{'s': 'Hello World'}`
- Output: `5`

### Example 2
- Input: `{'s': ' fly me to the moon '}`
- Output: `4`

### Example 3
- Input: `{'s': 'luffy is still joyboy'}`
- Output: `6`
