# 49. Group Anagrams

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/group-anagrams/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a list of strings:

```text
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
```

Group together the strings that are anagrams of each other.

Two strings are anagrams when they contain exactly the same characters with exactly the same frequencies.

For example:

```text
"eat"
"tea"
"ate"
```

are all anagrams because each one contains:

```text
e: 1
a: 1
t: 1
```

The order is different, but the multiset of characters is identical.

By contrast:

```text
"tan"
"bat"
```

are not anagrams because their character counts differ.

So the real problem is:

> Partition the input strings into groups where every string in the same group has the same character multiset.

The output order usually does not matter. The order of groups and the order inside each group are not important unless a particular test harness asks for normalized comparison.

---

### 2. Start From the Brute Force Baseline

The most direct approach is to compare strings pair by pair.

For each string, we could search through groups we have already built and ask:

```text
Is this string an anagram of the representative string for this group?
```

Conceptually:

```python
groups = []

for word in strs:
    placed = False

    for group in groups:
        representative = group[0]

        if word and representative are anagrams:
            group.append(word)
            placed = True
            break

    if not placed:
        groups.append([word])
```

This is correct if the anagram test is correct.

But it is inefficient because a word may be compared against many existing groups. If there are `n` strings, we may perform many repeated comparisons, and each comparison must inspect the characters of the strings.

The deeper question is:

> Can we compute one stable identity for each word so that all anagrams automatically land in the same bucket?

That identity is called a canonical key.

---

### 3. The Key Observation: Anagrams Have the Same Canonical Form

An anagram changes only character order.

It does not change character counts.

That means two words are anagrams if and only if they share a representation that ignores order but preserves multiplicity.

For example:

```text
"eat" -> sorted characters -> "aet"
"tea" -> sorted characters -> "aet"
"ate" -> sorted characters -> "aet"
```

All three words produce the same key:

```text
"aet"
```

But:

```text
"tan" -> "ant"
"nat" -> "ant"
"bat" -> "abt"
```

So `"tan"` and `"nat"` belong together, while `"bat"` belongs somewhere else.

This turns the problem from repeated pairwise comparison into direct grouping:

```text
canonical key -> all original strings with that key
```

That is exactly what a hash table is good at.

---

### 4. What Is a Canonical Key?

A canonical key is a representation with this property:

```text
word1 and word2 are anagrams
if and only if
key(word1) == key(word2)
```

This is the central invariant of the problem.

The key must satisfy both directions:

1. If two words are anagrams, they must produce the same key.
2. If two words produce the same key, they must be anagrams.

Sorting characters satisfies this invariant.

Why?

If two words are anagrams, they contain the same letters with the same counts. Sorting both words arranges those same letters into the same order, so the sorted strings match.

If two sorted strings match, then they contain the same sequence of letters after sorting. Therefore, the original words must have had the same character counts, so they are anagrams.

For lowercase English letters, a frequency-count tuple is another valid canonical key:

```text
"eat" -> (1 a, 0 b, 0 c, ..., 1 e, ..., 1 t, ...)
```

In Python that might be represented as a tuple of 26 integers.

Both approaches are valid:

- Sorted-string key: simpler to write and easy to understand.
- Count-tuple key: avoids sorting each word and can be faster when the alphabet is fixed and small.

The tutorial can use the sorted key because it expresses the invariant very clearly.

---

### 5. Hash Table Invariant

Maintain a dictionary:

```text
groups_by_key[key] = list of original words whose canonical key is key
```

The invariant is:

```text
After processing some prefix of strs, every processed word is stored in exactly one bucket,
and that bucket is indexed by the word's canonical anagram key.
```

For example, after processing:

```text
["eat", "tea", "tan"]
```

the table is:

```text
{
  "aet": ["eat", "tea"],
  "ant": ["tan"]
}
```

When the next word arrives, the algorithm does not need to compare it with every previous word.

It computes the word's key once and appends the word to exactly one bucket.

This is the entire power of the solution:

```text
same anagram identity -> same hash-table key -> same output group
```

---

### 6. Detailed Algorithm

Use a dictionary whose keys are canonical forms and whose values are lists of strings.

For each `word` in `strs`:

1. Compute the canonical key.
   - With sorting: `key = ''.join(sorted(word))`
   - With counts: `key = tuple(character_counts)`
2. Look up `key` in the dictionary.
3. Append the original `word` to the list stored at that key.
4. After all words are processed, return all dictionary values.

Important detail:

The bucket stores the original word, not the sorted key.

For example, `"eat"` is grouped under key `"aet"`, but the output should contain `"eat"`, not `"aet"`.

---

### 7. Detailed Example Walkthrough

Input:

```text
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
```

Start with an empty table:

```text
groups_by_key = {}
```

Process `"eat"`:

```text
sorted("eat") = "aet"
```

No bucket exists for `"aet"`, so create one:

```text
{
  "aet": ["eat"]
}
```

Process `"tea"`:

```text
sorted("tea") = "aet"
```

The bucket already exists, so append:

```text
{
  "aet": ["eat", "tea"]
}
```

Process `"tan"`:

```text
sorted("tan") = "ant"
```

Create a new bucket:

```text
{
  "aet": ["eat", "tea"],
  "ant": ["tan"]
}
```

Process `"ate"`:

```text
sorted("ate") = "aet"
```

Append to the `"aet"` bucket:

```text
{
  "aet": ["eat", "tea", "ate"],
  "ant": ["tan"]
}
```

Process `"nat"`:

```text
sorted("nat") = "ant"
```

Append to the `"ant"` bucket:

```text
{
  "aet": ["eat", "tea", "ate"],
  "ant": ["tan", "nat"]
}
```

Process `"bat"`:

```text
sorted("bat") = "abt"
```

Create a new bucket:

```text
{
  "aet": ["eat", "tea", "ate"],
  "ant": ["tan", "nat"],
  "abt": ["bat"]
}
```

Return the values:

```text
[
  ["eat", "tea", "ate"],
  ["tan", "nat"],
  ["bat"]
]
```

Any ordering of these groups is acceptable for the LeetCode problem.

---

### 8. Pseudocode

Using a sorted-string key:

```python
def group_anagrams(strs):
    groups_by_key = {}

    for word in strs:
        key = ''.join(sorted(word))

        if key not in groups_by_key:
            groups_by_key[key] = []

        groups_by_key[key].append(word)

    return list(groups_by_key.values())
```

In Python, `defaultdict(list)` makes the bucket creation shorter:

```python
from collections import defaultdict


def group_anagrams(strs):
    groups_by_key = defaultdict(list)

    for word in strs:
        key = ''.join(sorted(word))
        groups_by_key[key].append(word)

    return list(groups_by_key.values())
```

Using a frequency tuple for lowercase English letters:

```python
from collections import defaultdict


def group_anagrams(strs):
    groups_by_key = defaultdict(list)

    for word in strs:
        counts = [0] * 26

        for char in word:
            index = ord(char) - ord('a')
            counts[index] += 1

        key = tuple(counts)
        groups_by_key[key].append(word)

    return list(groups_by_key.values())
```

The tuple conversion matters because Python lists are mutable and cannot be used as dictionary keys.

---

### 9. Correctness

We prove the sorted-key algorithm returns exactly the required anagram groups.

#### Lemma 1: Anagrams produce the same key.

If two strings are anagrams, they contain exactly the same characters with exactly the same frequencies. Sorting both strings places that same multiset of characters into the same deterministic order. Therefore, both strings produce the same sorted key.

#### Lemma 2: Strings with the same key are anagrams.

If two strings have the same sorted key, then their sorted character sequences are identical. Therefore, each character appears the same number of times in both strings. So the original strings are anagrams.

#### Lemma 3: Every bucket contains only anagrams.

The algorithm places a word into the bucket indexed by its sorted key. If two words are in the same bucket, they have the same key. By Lemma 2, they are anagrams.

#### Lemma 4: All anagrams are placed in the same bucket.

If two words are anagrams, then by Lemma 1 they produce the same key. The algorithm appends both words to the bucket for that key, so they are placed in the same group.

#### Theorem: The algorithm returns exactly the correct grouping of anagrams.

Every output bucket contains only mutually anagrammatic words by Lemma 3. Every pair of anagrammatic input words appears in the same bucket by Lemma 4. Every input word is processed once and appended to one bucket. Therefore, the returned dictionary values form exactly the required partition of the input strings into anagram groups.

---

### 10. Complexity

Let:

```text
n = number of strings
k = maximum length of a string
```

For the sorted-key approach:

- Sorting one word costs `O(k log k)`.
- Processing all words costs `O(n * k log k)`.
- The hash table stores all original strings across its buckets, plus one key per group.
- Auxiliary space is `O(n * k)` if counting stored output strings and keys, or `O(n)` buckets plus key storage depending on how space is measured.

For the count-tuple approach with only lowercase English letters:

- Counting one word costs `O(k)`.
- Building the 26-length tuple costs `O(26)`, which is constant.
- Processing all words costs `O(n * k)`.
- Auxiliary space is still proportional to the stored groups and keys.

The sorted-key approach is often preferred in interviews because it is concise and hard to get wrong. The count-tuple approach is a useful optimization when the alphabet is fixed and known.

---

### 11. Common Pitfalls

- Returning the keys instead of the original strings. The sorted key is only for grouping; the output groups must contain the original input words.
- Using a non-canonical key, such as the first or last character. Anagrams can start and end with different characters.
- Ignoring duplicate character counts. `"abb"` and `"ab"` are not anagrams even though they use similar letters.
- Using a mutable list as a dictionary key for the count-based approach. Convert counts to a tuple first.
- Assuming the output order must match a specific order. LeetCode accepts any valid group ordering.
- Forgetting the empty string case. The key for `""` is also `""`, so all empty strings belong together.
- Accidentally grouping by a set of characters. A set loses multiplicity, so `"aab"` and `"ab"` would wrongly look the same.

---

### 12. First-Principles Summary

The problem is not really about comparing every string with every other string.

It is about finding the information that completely determines whether two strings belong together.

For anagrams, that information is the character multiset.

A canonical key is a hashable representation of that multiset:

```text
sorted characters
or
fixed alphabet frequency tuple
```

Once every word can be reduced to this key, grouping becomes mechanical:

```text
compute key -> append original word to bucket for key
```

The invariant is simple and powerful:

```text
Each bucket contains exactly the words with one canonical anagram identity.
```

When all words have been processed, the buckets themselves are the answer.

## Implementation
See `solutions/hash_table/p049_group_anagrams.py`.

## Tests
See `tests/hash_table/test_p049_group_anagrams.py`.

## Examples

### Example 1
- Input: `{'raw': '["eat","tea","tan","ate","nat","bat"]\n[""]\n["a"]'}`
- Output: `'See official examples'`

For the first official input, one valid grouping is:

```text
[["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

The groups may appear in a different order.

For the second official input:

```text
strs = [""]
```

the answer is:

```text
[[""]]
```

The empty string has an empty canonical key.

For the third official input:

```text
strs = ["a"]
```

the answer is:

```text
[["a"]]
```

## Follow-up Practice

- Trace the dictionary after every word in `['eat', 'tea', 'tan', 'ate', 'nat', 'bat']`.
- Explain why sorted characters are a valid canonical key.
- Replace the sorted key with a 26-count tuple and compare the complexity.
- Test duplicate words, such as `['a', 'a']`, and verify both copies remain in the output.
- Test strings with repeated letters, such as `['abb', 'bab', 'bba', 'ab']`, to confirm counts are preserved.
