# 290. Word Pattern

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/word-pattern/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
pattern = a string of pattern symbols
s       = a sentence containing words separated by single spaces
```

Return `true` if the words in `s` follow exactly the same pattern as the characters in `pattern`.

For example:

```text
pattern = "abba"
s       = "dog cat cat dog"
```

Line the two sequences up by position:

```text
pattern: a    b    b    a
words:   dog  cat  cat  dog
```

The first `a` corresponds to `dog`, and the last `a` also corresponds to `dog`.
The first `b` corresponds to `cat`, and the second `b` also corresponds to `cat`.

So this input is valid.

Now compare:

```text
pattern = "abba"
s       = "dog cat cat fish"
```

Line them up:

```text
pattern: a    b    b    a
words:   dog  cat  cat  fish
```

The first `a` maps to `dog`, but the last `a` maps to `fish`.
That is inconsistent, so this input is invalid.

The real problem is:

> Can every pattern character be paired with exactly one word, and can every word be paired with exactly one pattern character, consistently across all positions?

This is not only about checking whether repeated pattern characters repeat the same word.
It is also about checking the reverse direction: two different pattern characters must not share the same word.

---

### 2. The Input Is Two Parallel Sequences

The pattern is already a sequence of characters:

```text
"abba" -> ['a', 'b', 'b', 'a']
```

The sentence must be split into words:

```text
"dog cat cat dog" -> ['dog', 'cat', 'cat', 'dog']
```

After splitting, the problem becomes a position-by-position comparison between two sequences:

```text
pattern_chars[i] should correspond to words[i]
```

Before worrying about mappings, there is one simple necessary condition:

```text
len(pattern) == len(words)
```

If the lengths differ, there is no way for each pattern character to match exactly one word.

For example:

```text
pattern = "abba"
s       = "dog cat cat"
```

There are four pattern characters but only three words, so the answer must be `false` immediately.

---

### 3. Start From the Brute Force Baseline

A very direct way to solve the problem is to compare every pair of positions.

For any two indices `i` and `j`:

```text
pattern[i] == pattern[j]
```

should be true exactly when:

```text
words[i] == words[j]
```

Why?

If two positions have the same pattern character, they must have the same word.
If two positions have different pattern characters, they must have different words.

So the brute-force rule is:

```python
words = s.split()

if len(pattern) != len(words):
    return False

for i in range(len(pattern)):
    for j in range(len(pattern)):
        same_pattern = pattern[i] == pattern[j]
        same_word = words[i] == words[j]

        if same_pattern != same_word:
            return False

return True
```

This is correct because it checks the full structural relationship between all positions.

But it is inefficient.

There are `n` positions, so there are `O(n^2)` pairs of positions.
For an easy problem this may still pass for small inputs, but it misses the simpler idea hiding underneath:

> We do not need to compare every old position again. We only need to remember the mapping decisions we have already made.

---

### 4. The Key Observation: A Pattern Is a Mapping

When we see this pair:

```text
a -> dog
```

we are making a promise:

```text
Every future 'a' must also map to 'dog'.
```

When we later see:

```text
a -> fish
```

that violates the promise.

So one direction of the relationship can be stored as a hash table:

```text
pattern character -> word
```

For the valid example:

```text
pattern = "abba"
words   = ["dog", "cat", "cat", "dog"]
```

we eventually store:

```text
a -> dog
b -> cat
```

That table catches this invalid case:

```text
pattern = "abba"
words   = ["dog", "cat", "cat", "fish"]
```

because the last pair says:

```text
a -> fish
```

but the table already says:

```text
a -> dog
```

However, one table is not enough.

---

### 5. Why One Direction Is Not Enough

Consider:

```text
pattern = "ab"
s       = "dog dog"
```

If we only store `pattern character -> word`, we see:

```text
a -> dog
b -> dog
```

Each character is internally consistent:

```text
a always maps to dog
b always maps to dog
```

So a one-direction check would incorrectly accept the input.

But the pattern does not allow this. Different pattern characters represent different pattern symbols, so they must not collapse onto the same word.

The word `dog` cannot represent both `a` and `b`.

That means we also need the reverse promise:

```text
Every word maps back to exactly one pattern character.
```

Store another hash table:

```text
word -> pattern character
```

Now the invalid example is caught:

```text
dog -> a
```

is already stored, so when we try:

```text
dog -> b
```

there is a conflict.

---

### 6. The Bijection Invariant

The relationship must be a **bijection** between the set of pattern characters used and the set of words used.

A bijection means two things at the same time:

```text
1. Each pattern character maps to at most one word.
2. Each word maps to at most one pattern character.
```

Together, these prevent both kinds of mistakes:

```text
same pattern character -> two different words     invalid
same word              -> two different characters invalid
```

During the scan, maintain this invariant:

```text
For every processed position k:
pattern[k] and words[k] agree with both maps:

char_to_word[pattern[k]] == words[k]
word_to_char[words[k]] == pattern[k]
```

At a new position `i`, with:

```text
ch = pattern[i]
word = words[i]
```

there are four possible situations:

1. `ch` has been seen and maps to this same `word`: okay.
2. `ch` has been seen but maps to a different word: invalid.
3. `word` has been seen and maps to this same `ch`: okay.
4. `word` has been seen but maps to a different character: invalid.

If neither side has been seen before, we can safely create both mappings:

```text
ch -> word
word -> ch
```

This is the central invariant of the problem.

---

### 7. Detailed Algorithm

1. Split the sentence into words:

   ```python
   words = s.split()
   ```

2. If the number of words does not equal the number of pattern characters, return `False`.

3. Create two empty hash tables:

   ```text
   char_to_word = {}
   word_to_char = {}
   ```

4. Scan `pattern` and `words` together from left to right.

5. For each pair `(ch, word)`:

   - If `ch` already has a mapped word, that mapped word must equal `word`.
   - If `word` already has a mapped character, that mapped character must equal `ch`.
   - If either check fails, return `False`.
   - Otherwise, record both mappings.

6. If the scan finishes without conflicts, return `True`.

The algorithm does not guess a mapping in advance.
It discovers the only possible mapping forced by the first occurrence of each character and each word, then checks that every later occurrence respects it.

---

### 8. Pseudocode

```python
def wordPattern(pattern: str, s: str) -> bool:
    words = s.split()

    if len(pattern) != len(words):
        return False

    char_to_word = {}
    word_to_char = {}

    for ch, word in zip(pattern, words):
        if ch in char_to_word and char_to_word[ch] != word:
            return False

        if word in word_to_char and word_to_char[word] != ch:
            return False

        char_to_word[ch] = word
        word_to_char[word] = ch

    return True
```

An equivalent compact formulation is to assign first-seen identifiers to characters and words and compare the identifier sequence, but the two-map version is usually easier to reason about from first principles because it names the bijection directly.

---

### 9. Walkthrough: Valid Example

Input:

```text
pattern = "abba"
s       = "dog cat cat dog"
```

After splitting:

```text
words = ["dog", "cat", "cat", "dog"]
```

The lengths match:

```text
len(pattern) = 4
len(words)   = 4
```

Start with empty maps:

```text
char_to_word = {}
word_to_char = {}
```

#### Position 0

```text
ch = 'a'
word = "dog"
```

Neither has been seen before, so record:

```text
a -> dog
dog -> a
```

Maps:

```text
char_to_word = { a: dog }
word_to_char = { dog: a }
```

#### Position 1

```text
ch = 'b'
word = "cat"
```

Neither has been seen before, so record:

```text
b -> cat
cat -> b
```

Maps:

```text
char_to_word = { a: dog, b: cat }
word_to_char = { dog: a, cat: b }
```

#### Position 2

```text
ch = 'b'
word = "cat"
```

Check the existing promises:

```text
b -> cat   matches
cat -> b   matches
```

No conflict.

#### Position 3

```text
ch = 'a'
word = "dog"
```

Check the existing promises:

```text
a -> dog   matches
dog -> a   matches
```

No conflict.

The scan finishes, so the answer is:

```text
True
```

---

### 10. Walkthrough: Same Character, Different Word

Input:

```text
pattern = "abba"
s       = "dog cat cat fish"
```

The first three positions build the same mappings as before:

```text
a -> dog
b -> cat
```

At the last position:

```text
ch = 'a'
word = "fish"
```

But `a` already maps to `dog`:

```text
char_to_word['a'] = "dog"
```

The new word is `fish`, so this would require:

```text
a -> dog
and
a -> fish
```

A single pattern character cannot map to two words.
Return `False`.

---

### 11. Walkthrough: Different Characters, Same Word

Input:

```text
pattern = "ab"
s       = "dog dog"
```

At position 0:

```text
a -> dog
dog -> a
```

At position 1:

```text
ch = 'b'
word = "dog"
```

The character `b` has not been seen, so the forward map alone looks harmless.

But the reverse map says:

```text
dog -> a
```

The new pair would require:

```text
dog -> b
```

The same word cannot represent two different pattern characters.
Return `False`.

This is the example that explains why the reverse map is necessary.

---

### 12. Correctness

We prove that the algorithm returns `True` exactly when `s` follows `pattern`.

#### Lemma 1: If the algorithm returns `False`, the pattern relation is invalid.

The algorithm can return `False` in three cases.

First, if the lengths differ, at least one pattern character has no corresponding word or at least one word has no corresponding pattern character. Therefore the sequences cannot match position by position.

Second, if `char_to_word[ch]` already exists but is not equal to the current `word`, then the same pattern character appears with two different words. That violates the requirement that each pattern character represent exactly one word.

Third, if `word_to_char[word]` already exists but is not equal to the current `ch`, then the same word appears with two different pattern characters. That violates the requirement that each word represent exactly one pattern character.

So every `False` result is caused by a real violation.

#### Lemma 2: After each processed position, both maps correctly describe all processed pairs.

At a position `(ch, word)`, the algorithm first checks whether either existing mapping conflicts.
If a conflict exists, it returns `False` and stops.
If no conflict exists, assigning:

```text
char_to_word[ch] = word
word_to_char[word] = ch
```

records the current pair without contradicting any previous pair.

Therefore, after processing each position, every processed character-word pair agrees with both maps.

#### Lemma 3: If the algorithm finishes and returns `True`, the full mapping is a bijection.

By Lemma 2, every processed occurrence agrees with `char_to_word`, so no pattern character maps to two different words.
Also by Lemma 2, every processed occurrence agrees with `word_to_char`, so no word maps to two different pattern characters.

Thus the mapping is one-to-one in both directions over all positions.
That is exactly the required bijection between used pattern characters and used words.

#### Theorem: The algorithm is correct.

If the algorithm returns `False`, Lemma 1 shows the input cannot follow the pattern.
If the algorithm returns `True`, Lemma 3 shows the input has a valid bijection between pattern characters and words, so the sentence follows the pattern.

Therefore, the algorithm returns the correct answer.

---

### 13. Complexity

Let:

```text
n = len(pattern)
m = number of words in s
```

Splitting the sentence costs linear time in the length of `s`.
The scan compares `n` character-word pairs when the lengths match.
Each hash table lookup and assignment is expected `O(1)`.

So the time complexity is:

```text
O(len(s) + n)
```

Often this is described as simply:

```text
O(n)
```

where `n` is the total input size.

The auxiliary space is:

```text
O(k)
```

where `k` is the number of distinct pattern characters plus distinct words stored in the maps.
In the worst case, `k` is linear in the number of positions, so:

```text
O(n)
```

---

### 14. Common Pitfalls

#### Pitfall 1: Checking only `pattern character -> word`

This misses cases like:

```text
pattern = "ab"
s       = "dog dog"
```

The forward mappings are individually consistent, but the relationship is not one-to-one.
You need the reverse direction too.

#### Pitfall 2: Forgetting the length check

This input cannot be valid:

```text
pattern = "abba"
s       = "dog cat cat"
```

Even if the first three pairs look consistent, the fourth pattern character has no word.
Always check lengths after splitting.

#### Pitfall 3: Comparing characters with raw sentence positions

The pattern aligns with words, not characters of `s`.
The string:

```text
"dog cat cat dog"
```

has spaces and multi-character words, so use `s.split()` before comparing.

#### Pitfall 4: Treating repeated words as harmless extras

A repeated word is only allowed if it corresponds to the same repeated pattern character.
For example:

```text
pattern = "aaaa"
s       = "dog dog dog dog"
```

is valid, but:

```text
pattern = "abba"
s       = "dog dog dog dog"
```

is invalid because `a` and `b` would both map to `dog`.

#### Pitfall 5: Updating before checking without preserving consistency

It is safest to check both maps for conflicts first, then write the pair.
Overwriting an old mapping can hide the exact contradiction you needed to detect.

---

### 15. First-Principles Summary

The problem is about structural sameness between two sequences.

The pattern sequence and the word sequence follow the same structure only when equality relationships match:

```text
pattern[i] == pattern[j]
```

if and only if:

```text
words[i] == words[j]
```

The brute-force version checks that rule for every pair of positions.
The efficient version stores the same information as a bijection:

```text
pattern character -> word
word -> pattern character
```

The first occurrence of a character or word creates a promise.
Every later occurrence must keep that promise.
If any position breaks either direction of the promise, the input is invalid.
If every position preserves both directions, the sentence follows the pattern.

## Implementation

See `solutions/hash_table/p290_word_pattern.py`.

## Tests

See `tests/hash_table/test_p290_word_pattern.py`.

## Examples

### Example 1

```text
Input: pattern = "abba", s = "dog cat cat dog"
Output: true
Explanation: 'a' maps to "dog" and 'b' maps to "cat".
```

### Example 2

```text
Input: pattern = "abba", s = "dog cat cat fish"
Output: false
Explanation: 'a' would need to map to both "dog" and "fish".
```

### Example 3

```text
Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false
Explanation: 'a' would need to map to multiple different words.
```

### Reverse-Direction Example

```text
Input: pattern = "ab", s = "dog dog"
Output: false
Explanation: "dog" would need to map back to both 'a' and 'b'.
```
