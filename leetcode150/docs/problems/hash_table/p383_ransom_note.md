# 383. Ransom Note

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/ransom-note/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two strings:

```text
ransomNote
magazine
```

You want to know whether the characters in `magazine` are enough to build `ransomNote`.

Each character in `magazine` is a physical resource, like a cut-out letter from a page. If `magazine` contains one `a`, then you can use that `a` once. After it is used, it is gone.

So the question is not:

> Does every character type in `ransomNote` appear somewhere in `magazine`?

That would ignore duplicates.

The real question is:

> For every character, does `magazine` contain at least as many copies as `ransomNote` needs?

For example:

```text
ransomNote = "aa"
magazine   = "ab"
```

The character `a` appears in both strings, but `ransomNote` needs two copies of `a` while `magazine` provides only one. Therefore the answer is `False`.

In contrast:

```text
ransomNote = "aa"
magazine   = "aab"
```

Now `magazine` provides two copies of `a`, so the answer is `True`.

This problem is fundamentally about **limited resources**.

### 2. Start From the Brute Force Baseline

A direct way to simulate building the note is:

1. For each character needed by `ransomNote`, search for that character in `magazine`.
2. If you find an unused copy, consume it.
3. If you cannot find one, return `False`.
4. If every ransom-note character is matched, return `True`.

Conceptually:

```python
used = [False] * len(magazine)

for needed in ransomNote:
    found = False

    for i, available in enumerate(magazine):
        if not used[i] and available == needed:
            used[i] = True
            found = True
            break

    if not found:
        return False

return True
```

This is correct because it respects the one-use-only rule: once a magazine character is marked used, it cannot be reused.

But it is inefficient. If `ransomNote` has length `r` and `magazine` has length `m`, then in the worst case we may scan most of `magazine` for every character in `ransomNote`.

That gives:

```text
O(r * m) time
```

The repeated work is obvious: we keep asking the same kind of question again and again:

> How many unused copies of this character are still available?

Instead of searching the magazine repeatedly, we can count the resources once.

### 3. Key Observation: Order Does Not Matter

The word "construct" may sound like we must build `ransomNote` in order. But the order of characters in `magazine` is irrelevant.

If:

```text
ransomNote = "abc"
magazine   = "cba"
```

The answer is still `True`. We can cut out `c`, `b`, and `a` from anywhere in the magazine and arrange them into `abc`.

So the only information that matters is frequency:

```text
How many times does each character appear?
```

Not positions.
Not adjacency.
Not substrings.
Not relative order.

This is why a hash table is the natural tool: it maps each character to the number of available copies.

```text
character -> remaining count
```

### 4. The Frequency / Resource Invariant

The clean invariant is:

```text
After processing some prefix of ransomNote,
counts[c] equals the number of unused copies of character c still available from magazine.
```

At the beginning, before using any ransom-note character, `counts` stores all resources from `magazine`.

For example:

```text
magazine = "aab"

counts = {
  'a': 2,
  'b': 1
}
```

Now process `ransomNote = "aa"`.

Before using the first `a`:

```text
counts['a'] = 2
```

There is at least one available `a`, so consume one:

```text
counts['a'] = 1
```

Before using the second `a`:

```text
counts['a'] = 1
```

There is still one available `a`, so consume it:

```text
counts['a'] = 0
```

The note is complete, so return `True`.

The invariant is powerful because it turns the entire problem into one local decision per character:

```text
If the needed character has positive remaining count, consume it.
Otherwise construction is impossible.
```

### 5. Detailed Algorithm

Use two passes.

#### Pass 1: Count the Magazine Characters

Create an empty frequency table.

For each character `ch` in `magazine`:

```text
counts[ch] += 1
```

After this pass, `counts` represents the full supply of letters.

#### Pass 2: Spend Characters on the Ransom Note

For each character `ch` in `ransomNote`:

1. Check how many copies of `ch` remain.
2. If the remaining count is zero or missing, return `False`.
3. Otherwise decrement the count because one copy has been used.

If the loop finishes, every required character was successfully paid for, so return `True`.

The algorithm is intentionally asymmetric:

```text
magazine    -> supply
ransomNote  -> demand
```

We count supply first, then subtract demand.

### 6. Example Walkthrough

Use the third official example:

```text
ransomNote = "aa"
magazine   = "aab"
```

First count the magazine.

Start:

```text
counts = {}
```

Read first `a`:

```text
counts = {'a': 1}
```

Read second `a`:

```text
counts = {'a': 2}
```

Read `b`:

```text
counts = {'a': 2, 'b': 1}
```

Now process the ransom note.

Need first `a`:

```text
counts['a'] = 2
```

There is an available `a`, so consume one:

```text
counts = {'a': 1, 'b': 1}
```

Need second `a`:

```text
counts['a'] = 1
```

There is still an available `a`, so consume it:

```text
counts = {'a': 0, 'b': 1}
```

The ransom note has no more characters. Return `True`.

The leftover `b` does not matter. Extra magazine characters are allowed because the problem asks whether the note can be constructed, not whether every magazine character must be used.

### 7. Failure Walkthrough

Use the second official example:

```text
ransomNote = "aa"
magazine   = "ab"
```

Count the magazine:

```text
counts = {'a': 1, 'b': 1}
```

Need first `a`:

```text
counts['a'] = 1
```

Consume it:

```text
counts = {'a': 0, 'b': 1}
```

Need second `a`:

```text
counts['a'] = 0
```

There are no unused `a` characters left. Return `False` immediately.

This is the resource invariant catching the exact reason construction fails.

### 8. Code / Pseudocode

A direct implementation is:

```python
def canConstruct(ransomNote: str, magazine: str) -> bool:
    counts = {}

    for ch in magazine:
        counts[ch] = counts.get(ch, 0) + 1

    for ch in ransomNote:
        if counts.get(ch, 0) == 0:
            return False
        counts[ch] -= 1

    return True
```

In Python, `collections.Counter` can express the same idea more compactly, but the manual dictionary version is often better for understanding the invariant:

```python
from collections import Counter

def canConstruct(ransomNote: str, magazine: str) -> bool:
    available = Counter(magazine)

    for ch in ransomNote:
        if available[ch] == 0:
            return False
        available[ch] -= 1

    return True
```

Both versions implement the same rule:

```text
Never spend a character unless at least one unused copy remains.
```

### 9. Correctness Argument

We prove the algorithm returns `True` exactly when `ransomNote` can be constructed from `magazine`.

#### Invariant

After processing the first `k` characters of `ransomNote`, for every character `c`:

```text
counts[c] = number of copies of c in magazine
            minus
            number of copies of c used by the first k ransom-note characters
```

Equivalently, `counts[c]` is the remaining unused supply of `c`.

#### Initialization

Before any ransom-note character is processed, no characters have been used. The first pass counts every character in `magazine`, so `counts[c]` is exactly the initial supply of `c`.

Therefore the invariant holds for `k = 0`.

#### Maintenance

Assume the invariant holds before processing the next ransom-note character `ch`.

If `counts[ch] == 0`, then there is no unused copy of `ch` left in the magazine. Since the next character of the note requires `ch`, no valid construction is possible. Returning `False` is correct.

If `counts[ch] > 0`, then at least one unused copy of `ch` exists. The algorithm consumes one by decrementing `counts[ch]`. All other character counts remain unchanged. Therefore the table again represents exactly the remaining unused supply after processing one more character of `ransomNote`.

So the invariant is preserved.

#### Termination

If the algorithm finishes the ransom-note loop, every character in `ransomNote` was processed and each one successfully consumed an available magazine character.

Thus there exists an assignment of magazine characters to all ransom-note characters, with no magazine character used more than once. Therefore `ransomNote` can be constructed, and returning `True` is correct.

Combining the failure and success cases, the algorithm is correct.

### 10. Complexity

Let:

```text
r = len(ransomNote)
m = len(magazine)
```

Counting the magazine scans `m` characters.

Spending characters for the ransom note scans `r` characters.

So the time complexity is:

```text
O(m + r)
```

The hash table stores one entry per distinct character in `magazine`.

So the auxiliary space complexity is:

```text
O(u)
```

where `u` is the number of distinct characters in `magazine`.

If the input is known to contain only lowercase English letters, then `u <= 26`, so the space can be considered `O(1)` under that fixed alphabet assumption. Without relying on a fixed alphabet, the general statement is `O(u)`.

### 11. Common Pitfalls

- Checking only membership instead of counts. `"aa"` cannot be built from `"ab"`, even though `a` appears in the magazine.
- Forgetting to decrement after using a character. That accidentally allows the same magazine character to be reused many times.
- Treating magazine order as important. The magazine is a pool of characters, not a sequence that must match the note.
- Requiring all magazine characters to be used. Extra characters in `magazine` are harmless.
- Building counts from `ransomNote` and then forgetting to compare against `magazine` correctly. Either direction can work, but the meaning of the table must stay clear.
- Returning `False` too late. As soon as a needed count is unavailable, construction is impossible.

### 12. First-Principles Summary

The problem is a resource-allocation question.

Each character in `magazine` is one consumable unit. Each character in `ransomNote` is one unit of demand. A construction is possible exactly when every demand can be matched to a distinct available supply unit of the same character.

A frequency table is the minimal state needed because order does not matter and duplicates do matter.

The invariant is:

```text
counts always stores the unused magazine resources remaining after satisfying the part of ransomNote already processed.
```

With that invariant, each step becomes simple:

```text
Need character ch.
If no ch remains, fail.
Otherwise spend one ch and continue.
```

That is the entire problem from first principles.

## Implementation
See `solutions/hash_table/p383_ransom_note.py`.

## Tests
See `tests/hash_table/test_p383_ransom_note.py`.

## Examples

### Example 1
- Input: `{'ransomNote': 'a', 'magazine': 'b'}`
- Output: `False`

### Example 2
- Input: `{'ransomNote': 'aa', 'magazine': 'ab'}`
- Output: `False`

### Example 3
- Input: `{'ransomNote': 'aa', 'magazine': 'aab'}`
- Output: `True`

## Follow-up Practice
- Explain why `"aa"` and `"ab"` fails even though both contain `a`.
- Trace the remaining-count table after every consumed character.
- Try cases with extra magazine characters, missing characters, and repeated required characters.
