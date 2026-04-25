# 242. Valid Anagram

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-anagram/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two strings `s` and `t`, determine whether `t` is an anagram of `s`.

An anagram is not about preserving order. It is about preserving the exact multiset of characters.

For example:

```text
s = "anagram"
t = "nagaram"
```

These strings are anagrams because each character appears the same number of times in both strings:

```text
a -> 3
n -> 1
g -> 1
r -> 1
m -> 1
```

The positions changed, but the inventory of letters did not.

By contrast:

```text
s = "rat"
t = "car"
```

They are not anagrams. Both have length `3`, but their character inventories differ:

```text
s has r, a, t
t has c, a, r
```

So the problem is really asking:

> Do `s` and `t` contain exactly the same characters with exactly the same frequencies?

That means duplicates matter. It is not enough to know which distinct letters appear.

For example:

```text
s = "aab"
t = "ab"
```

Both strings use only `a` and `b`, but they are not anagrams because `s` has two `a` characters while `t` has one.

### 2. Start From the Brute Force Baseline

A direct way to test an anagram is to try to match every character in `s` with one unused equal character in `t`.

Conceptually:

```python
used = [False] * len(t)

for char_s in s:
    found_match = False

    for i, char_t in enumerate(t):
        if not used[i] and char_t == char_s:
            used[i] = True
            found_match = True
            break

    if not found_match:
        return False

return every character in t was used
```

This mirrors the definition: every character from `s` must be paired with a distinct identical character from `t`.

It is correct, but inefficient. In the worst case, each character in `s` may scan most of `t`, so the time cost is `O(n^2)` when the strings have length `n`.

A slightly cleaner brute-force variant is sorting:

```python
return sorted(s) == sorted(t)
```

Sorting works because anagrams become identical when characters are put into canonical order:

```text
sorted("anagram") = "aaagmnr"
sorted("nagaram") = "aaagmnr"
```

This is much simpler and usually fast enough, but sorting does more work than the problem fundamentally requires. We do not need the characters in order. We only need their counts.

Sorting costs `O(n log n)`. The first-principles question is:

> Can we compare the character inventories directly without arranging the characters?

### 3. The Key Observation: Order Is Noise, Counts Are Signal

An anagram may move characters around arbitrarily.

That means position cannot be the thing we compare:

```text
s[0] does not need to equal t[0]
s[1] does not need to equal t[1]
...
```

The stable information is frequency.

For every character `c`, an anagram must satisfy:

```text
count of c in s == count of c in t
```

If this equality holds for every character, then every copy of every character in `s` can be paired with one copy in `t`, and no extra character remains in `t`.

If it fails for even one character, the strings cannot be anagrams.

So we can reduce the entire problem to a frequency comparison.

### 4. The Frequency Invariant

The most useful invariant is a balance table:

```text
balance[c] = (# of c seen in s) - (# of c seen in t)
```

After processing both strings, the two strings are anagrams exactly when every balance is zero.

Why zero?

- `balance[c] > 0` means `s` has more copies of `c` than `t`.
- `balance[c] < 0` means `t` has more copies of `c` than `s`.
- `balance[c] == 0` means the two strings have the same number of `c` characters.

So the invariant captures the only information that matters: the remaining unmatched count for each character.

There are two common ways to maintain this invariant.

#### Option A: Build Two Count Tables

```text
count_s[c] = number of times c appears in s
count_t[c] = number of times c appears in t
```

Then compare the two dictionaries.

#### Option B: Build One Balance Table

```text
for c in s: balance[c] += 1
for c in t: balance[c] -= 1
```

Then check whether all balances are zero.

Both are based on the same invariant. The one-table version makes the cancellation idea explicit.

### 5. A Necessary Early Check: Length

If two strings have different lengths, they cannot be anagrams.

Why?

An anagram rearranges characters. Rearranging cannot create or delete characters.

So before counting anything:

```python
if len(s) != len(t):
    return False
```

This is not just a performance trick. It is a direct consequence of the definition.

It also prevents misleading partial matches such as:

```text
s = "ab"
t = "aab"
```

The shorter string cannot contain the same multiset of characters as the longer one.

### 6. Detailed Algorithm

Use one dictionary called `balance`.

1. If `s` and `t` have different lengths, return `False`.
2. Create an empty hash table `balance`.
3. For each character in `s`, increment that character's balance.
4. For each character in `t`, decrement that character's balance.
5. If every final balance is `0`, return `True`.
6. Otherwise, return `False`.

In Python-like pseudocode:

```python
def isAnagram(s, t):
    if len(s) != len(t):
        return False

    balance = {}

    for char in s:
        balance[char] = balance.get(char, 0) + 1

    for char in t:
        balance[char] = balance.get(char, 0) - 1

    for count in balance.values():
        if count != 0:
            return False

    return True
```

Because the problem constraints on LeetCode use lowercase English letters, an array of size `26` also works:

```python
def isAnagram(s, t):
    if len(s) != len(t):
        return False

    balance = [0] * 26

    for char in s:
        balance[ord(char) - ord('a')] += 1

    for char in t:
        balance[ord(char) - ord('a')] -= 1

    return all(count == 0 for count in balance)
```

The hash-table version is more general because it works for any hashable character set. The fixed-array version is more specialized and uses constant extra space under the lowercase-English-letter constraint.

### 7. Detailed Walkthrough: `s = "anagram"`, `t = "nagaram"`

Start with an empty balance table:

```text
balance = {}
```

Process `s = "anagram"` by adding counts.

After reading all of `s`:

```text
a: 3
n: 1
g: 1
r: 1
m: 1
```

This means `s` has three `a` characters and one each of `n`, `g`, `r`, and `m`.

Now process `t = "nagaram"` by subtracting counts.

Read `n`:

```text
n: 1 -> 0
```

The `n` from `t` cancels the `n` from `s`.

Read `a`:

```text
a: 3 -> 2
```

One of the three `a` characters has been matched.

Read `g`:

```text
g: 1 -> 0
```

Read `a` again:

```text
a: 2 -> 1
```

Read `r`:

```text
r: 1 -> 0
```

Read `a` again:

```text
a: 1 -> 0
```

Read `m`:

```text
m: 1 -> 0
```

Final balances:

```text
a: 0
n: 0
g: 0
r: 0
m: 0
```

Every character count cancels perfectly, so the strings are anagrams.

Return `True`.

### 8. Walkthrough: `s = "rat"`, `t = "car"`

Build balances from `s`:

```text
r: 1
a: 1
t: 1
```

Subtract characters from `t`.

Read `c`:

```text
c: 0 -> -1
```

This means `t` contains a `c` that `s` has not supplied.

Read `a`:

```text
a: 1 -> 0
```

Read `r`:

```text
r: 1 -> 0
```

Final balances:

```text
r: 0
a: 0
t: 1
c: -1
```

The `t: 1` means `s` has an unmatched `t`.

The `c: -1` means the second string has an extra `c`.

Because not every balance is zero, return `False`.

### 9. Correctness

We prove that the balance-table algorithm returns `True` exactly when `s` and `t` are anagrams.

#### Lemma 1: After processing `s`, `balance[c]` equals the number of occurrences of `c` in `s`.

The algorithm starts with every missing balance treated as `0`. Each time it sees character `c` in `s`, it increments `balance[c]` by `1`. No other operation during this first pass changes `balance[c]`. Therefore, after the first pass, `balance[c]` is exactly the count of `c` in `s`.

#### Lemma 2: After processing both strings, `balance[c]` equals `count_s(c) - count_t(c)` for every character `c`.

By Lemma 1, after the first pass `balance[c] = count_s(c)`. During the second pass, the algorithm subtracts `1` from `balance[c]` for each occurrence of `c` in `t`. After all occurrences in `t` have been processed, the total subtraction is `count_t(c)`. Therefore:

```text
balance[c] = count_s(c) - count_t(c)
```

#### Lemma 3: If every final balance is zero, then `s` and `t` are anagrams.

If every balance is zero, then for every character `c`:

```text
count_s(c) - count_t(c) = 0
```

So:

```text
count_s(c) = count_t(c)
```

That means the two strings contain exactly the same number of every character. Therefore one string can be rearranged into the other, so they are anagrams.

#### Lemma 4: If `s` and `t` are anagrams, then every final balance is zero.

If `s` and `t` are anagrams, they contain exactly the same number of every character. Therefore, for every `c`:

```text
count_s(c) = count_t(c)
```

Using Lemma 2:

```text
balance[c] = count_s(c) - count_t(c) = 0
```

So every final balance is zero.

#### Conclusion

By Lemma 3 and Lemma 4, the algorithm returns `True` if and only if the two strings are anagrams.

### 10. Complexity

Let `n = len(s)` and `m = len(t)`.

The early length check costs `O(1)`.

If the lengths differ, the algorithm returns immediately.

If the lengths are equal, let that common length be `n`.

- Building counts from `s` costs `O(n)`.
- Subtracting counts from `t` costs `O(n)`.
- Checking the final balances costs `O(k)`, where `k` is the number of distinct characters seen.

Since `k <= 2n`, the total time is:

```text
O(n)
```

The auxiliary space is:

```text
O(k)
```

where `k` is the number of distinct characters. Under the lowercase-English-letter constraint, `k <= 26`, so the auxiliary space is `O(1)` with respect to input length.

### 11. Common Pitfalls

#### Pitfall 1: Comparing Sets Instead of Counts

This is wrong:

```python
return set(s) == set(t)
```

It fails on duplicates:

```text
s = "aab"
t = "ab"
```

Both sets are `{a, b}`, but the strings are not anagrams.

#### Pitfall 2: Forgetting the Length Check

A balance-table implementation can still work without an explicit length check if it checks all final balances, but the length check is a simple necessary condition and prevents unnecessary counting.

It also makes the reasoning clearer:

```text
Different number of characters -> impossible to be a rearrangement
```

#### Pitfall 3: Only Checking Characters From One String Incorrectly

If you build counts from `s` and decrement using `t`, be careful not to ignore characters that appear only in `t`.

For example:

```text
s = "ab"
t = "ac"
```

The `c` must be detected as extra in `t`, and the `b` must be detected as missing from `t`.

A final all-zero balance check catches both.

#### Pitfall 4: Assuming Alphabet Constraints That Are Not Present

For LeetCode 242, lowercase English letters allow a `26`-element array.

But if the input can contain Unicode characters, uppercase letters, punctuation, or arbitrary symbols, a fixed `26`-element array is no longer enough. A hash table is the safer general solution.

#### Pitfall 5: Treating Sorting as the Only Natural Solution

Sorting is valid and concise, but it solves a stronger problem than necessary: it fully orders the characters.

For anagram checking, the only required information is frequency. Counting goes directly after that information and avoids the `log n` sorting factor.

### 12. First-Principles Summary

An anagram is a rearrangement, and rearrangement preserves exactly one essential thing: the multiplicity of each character.

So the problem should not be viewed as a positional comparison. It should be viewed as an inventory comparison.

The balance table records that inventory difference:

```text
balance[c] = copies of c from s not yet canceled by copies of c from t
```

When all balances are zero, every character supplied by `s` has been canceled by an identical character in `t`, and no extra character remains. That is precisely the definition of an anagram.

## Implementation

See `solutions/hash_table/p242_valid_anagram.py`.

## Tests

See `tests/hash_table/test_p242_valid_anagram.py`.

## Examples

### Example 1

- Input: `s = "anagram"`, `t = "nagaram"`
- Output: `true`
- Explanation: Both strings contain three `a` characters and one each of `n`, `g`, `r`, and `m`.

### Example 2

- Input: `s = "rat"`, `t = "car"`
- Output: `false`
- Explanation: `s` contains `t`, while `t` contains `c`; their character inventories are different.

## Follow-up Practice

- Trace the balance table for `s = "aacc"`, `t = "ccac"`.
- Explain why sorting works, then explain why counting is more direct.
- Decide whether a `26`-element array is safe for a given input constraint.
- Test duplicate-heavy cases such as `"aaab"` versus `"abaa"`.
