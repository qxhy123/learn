# 151. Reverse Words in a String

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/reverse-words-in-a-string/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: string, two-pointers, string-building

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a string `s`, return a new string where the words appear in reverse order.

A **word** is a maximal consecutive run of non-space characters. Spaces are not words. They only separate words.

For example:

```text
s = "the sky is blue"
```

The words are:

```text
["the", "sky", "is", "blue"]
```

Reversing the word order gives:

```text
["blue", "is", "sky", "the"]
```

Joining those words with exactly one space gives:

```text
"blue is sky the"
```

The output has two formatting requirements:

```text
No leading spaces.
No trailing spaces.
Exactly one space between adjacent words.
```

So the problem is not asking us to preserve the original spacing. It is asking us to find the words, reverse the word sequence, and emit the reversed sequence in normalized form.

That distinction is the center of the problem.

For:

```text
s = "  hello world  "
```

the useful data is:

```text
["hello", "world"]
```

The leading and trailing spaces do not appear in the answer. They only tell us that there is no word before `"hello"` and no word after `"world"`.

For:

```text
s = "a good   example"
```

the useful data is:

```text
["a", "good", "example"]
```

The three spaces between `"good"` and `"example"` still act as only one separator. They do not mean the answer should contain three spaces.

### 2. Start From the Brute Force Idea

A brute-force mental model is:

1. Try to build the final answer by scanning the original string from right to left.
2. Whenever a word is found, copy that word into the result.
3. Skip spaces.
4. Insert one space between copied words.

This can be made to work, but it is easy to make mistakes:

```text
Should we add a space before or after a copied word?
What happens when the input starts or ends with spaces?
What happens when there are several spaces between words?
How do we avoid an extra trailing space in the answer?
```

Another direct baseline is cleaner:

1. Scan the string from left to right.
2. Extract every word into a list.
3. Reverse the list.
4. Join the words with one space.

Conceptually:

```python
words = []

for each word in s:
    words.append(word)

words.reverse()
return " ".join(words)
```

This is already close to optimal. Any correct algorithm must inspect the input characters to know where the words are, so the lower bound is `O(n)` time for a string of length `n`.

In Python, the built-in tokenizer version is:

```python
return " ".join(reversed(s.split()))
```

Here, `s.split()` without an argument treats runs of whitespace as separators and drops leading/trailing whitespace. For this LeetCode problem, inputs use spaces, and that behavior matches the required normalized output.

The same idea can also be implemented manually with two pointers, which makes the state and invariants explicit.

### 3. Key Observation: Words Are Tokens, Spaces Are Separators

The important first-principles observation is:

> The output is determined only by the word tokens and their order, not by the original spaces.

These inputs all produce the same output:

```text
"the sky is blue"
"  the sky is blue"
"the   sky   is   blue   "
"   the     sky is      blue"
```

They all contain the same token sequence:

```text
["the", "sky", "is", "blue"]
```

So they all return:

```text
"blue is sky the"
```

This lets us separate the problem into two independent jobs:

```text
Tokenize: collect the words and discard separator spaces.
Emit: output the tokens in reverse order with one separator space.
```

Once we think in tokens, the messy spacing cases become ordinary cases. A leading space is just a separator before the first token. A trailing space is just a separator after the last token. Multiple spaces are still one separator boundary between two tokens.

### 4. State and Invariant

For a manual left-to-right scan, maintain:

```text
i      = current index in s
n      = len(s)
words  = complete words found so far, in original order
```

The scan invariant is:

```text
After processing all characters before index i,
words contains exactly the complete words that appear before i,
in their original left-to-right order,
and words contains no spaces.
```

This invariant is strong enough because, after the scan finishes:

```text
words contains every word in the input, in original order.
```

Then:

```text
reversed(words)
```

is exactly the desired word order, and:

```text
" ".join(reversed(words))
```

is exactly the desired spacing.

The scan has two local states:

```text
Skipping separators.
Reading a word.
```

When skipping separators, we advance while `s[i] == " "`.

When reading a word, we remember the start index and advance while `s[i] != " "`.

The transition is simple:

```text
spaces -> first non-space character starts a word
word   -> next space or end of string finishes the word
```

### 5. Detailed Algorithm

Manual tokenization algorithm:

1. Initialize `words = []`.
2. Initialize `i = 0`.
3. While `i < len(s)`:
4. Skip every space at the current position.
5. Mark the current index as `start`.
6. Move `i` forward until the next space or the end of the string.
7. If `start < i`, append `s[start:i]` to `words`.
8. After the scan, return `" ".join(reversed(words))`.

The `start < i` check matters when the scan skips spaces at the end of the string. In that case there is no word to append.

Python implementation:

```python
def reverse_words(s: str) -> str:
    words = []
    i = 0
    n = len(s)

    while i < n:
        while i < n and s[i] == " ":
            i += 1

        start = i

        while i < n and s[i] != " ":
            i += 1

        if start < i:
            words.append(s[start:i])

    return " ".join(reversed(words))
```

Equivalent high-level Python:

```python
def reverse_words(s: str) -> str:
    return " ".join(reversed(s.split()))
```

Both versions follow the same first-principles plan:

```text
extract words
reverse word order
normalize spaces while joining
```

### 6. Walkthrough: `"  hello world  "`

Input:

```text
s = "  hello world  "
```

The indices are:

```text
index:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
char:     h e l l o   w o r l  d
```

Start:

```text
i = 0
words = []
```

Skip leading spaces:

```text
i = 0 -> s[i] is space, skip
i = 1 -> s[i] is space, skip
i = 2 -> s[i] is "h", stop skipping
```

Read the first word:

```text
start = 2
advance through h, e, l, l, o
stop at index 7 because s[7] is space
```

Append:

```text
s[2:7] = "hello"
words = ["hello"]
```

Continue:

```text
i = 7 -> space, skip
i = 8 -> "w", stop skipping
```

Read the second word:

```text
start = 8
advance through w, o, r, l, d
stop at index 13 because s[13] is space
```

Append:

```text
s[8:13] = "world"
words = ["hello", "world"]
```

Skip trailing spaces:

```text
i = 13 -> space, skip
i = 14 -> space, skip
i = 15 -> end of string
```

Now all tokens have been collected:

```text
words = ["hello", "world"]
```

Reverse:

```text
["world", "hello"]
```

Join with one space:

```text
"world hello"
```

Notice that the two leading spaces and two trailing spaces never enter `words`, so they cannot leak into the output.

### 7. Walkthrough: `"a good   example"`

Input:

```text
s = "a good   example"
```

Start:

```text
i = 0
words = []
```

Read the first word:

```text
start = 0
advance through "a"
stop at index 1 because s[1] is space
words = ["a"]
```

Skip the separator:

```text
i = 1 -> space, skip
i = 2 -> "g", stop skipping
```

Read the second word:

```text
start = 2
advance through g, o, o, d
stop at index 6 because s[6] is space
words = ["a", "good"]
```

Skip the run of spaces:

```text
i = 6 -> space, skip
i = 7 -> space, skip
i = 8 -> space, skip
i = 9 -> "e", stop skipping
```

The three spaces are treated as one separator boundary. They do not create empty words.

Read the third word:

```text
start = 9
advance through e, x, a, m, p, l, e
stop at end of string
words = ["a", "good", "example"]
```

Reverse:

```text
["example", "good", "a"]
```

Join:

```text
"example good a"
```

The output has one space between each pair of words even though the input had three spaces between `"good"` and `"example"`.

### 8. Correctness

We prove the manual algorithm returns exactly the required string.

First, the scan records every word.

The outer loop advances through the string from left to right. Before reading a word, it skips all spaces. If a non-space character remains, that character is the first character of the next word. The second inner loop advances until the next space or the end of the string, so `s[start:i]` is exactly one maximal run of non-space characters. That is exactly one word by the problem definition.

Second, the scan records no non-words.

The algorithm appends only slices found by the word-reading loop. That loop starts at a non-space character and stops before a space or the end of the string. Therefore every appended slice contains only non-space characters. The `start < i` guard prevents appending an empty slice after trailing spaces.

Third, the recorded word order is the original left-to-right order.

The index `i` only moves forward. Each appended word is the next word encountered in the input, and no earlier index is revisited. Therefore `words` contains all input words in their original order.

After the scan, `reversed(words)` contains exactly the same words in reverse order. Joining with `" "` places exactly one space between adjacent words and no spaces before the first word or after the last word. That matches the required output format.

Therefore the algorithm is correct.

### 9. Complexity

Let `n = len(s)`.

Time complexity:

```text
O(n)
```

Each character is skipped or read a constant number of times during tokenization. Reversing the word order and joining the output also touches the total word characters once.

Space complexity:

```text
O(n)
```

The list of words and the returned string can together require linear space in the size of the input. In Python, strings are immutable, so constructing the output requires a new string.

### 10. Pitfalls

Common mistakes:

```text
Reversing characters instead of reversing words.
Preserving multiple spaces from the input.
Leaving a leading space in the output.
Leaving a trailing space in the output.
Treating consecutive spaces as empty words.
Appending an empty word after skipping trailing spaces.
Using split(" ") and forgetting that it produces empty strings for repeated spaces.
```

The safest rule is:

```text
Spaces are separators, not output data.
```

If using Python, prefer:

```python
s.split()
```

over:

```python
s.split(" ")
```

because `split()` groups whitespace separators and removes leading/trailing whitespace, while `split(" ")` preserves empty tokens caused by repeated spaces.

### 11. Summary

The problem becomes simple once the input is viewed as a sequence of word tokens separated by spaces.

The algorithm is:

```text
Collect the words.
Reverse the word list.
Join with exactly one space.
```

This directly enforces the required output format and avoids special-case handling for leading spaces, trailing spaces, or multiple spaces between words.

## Implementation

See `solutions/array_string/p151_reverse_words_in_a_string.py`.

## Tests

See `tests/array_string/test_p151_reverse_words_in_a_string.py`.

## Examples

- `{'s': 'the sky is blue'}` -> `'blue is sky the'`
- `{'s': '  hello world  '}` -> `'world hello'`
- `{'s': 'a good   example'}` -> `'example good a'`

## Follow-up Practice

- Solve the problem manually with a right-to-left scan that emits each discovered word into an output list.
- Solve the problem with the tokenize-then-reverse approach shown above.
- Compare `s.split()` with `s.split(" ")` on inputs with leading, trailing, and repeated spaces.
- Practice related string-token problems where separators should be normalized rather than preserved.
