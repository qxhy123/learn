# 205. Isomorphic Strings

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/isomorphic-strings/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two strings `s` and `t`, decide whether `s` can be transformed into `t` by replacing each character in `s` with exactly one character.

The replacement rule has two important parts:

```text
Every occurrence of the same character in s must map to the same character in t.
Different characters in s cannot map to the same character in t.
```

So the problem is not asking whether the two strings contain the same letters.

It is asking whether they have the same **shape of equality**.

For example:

```text
s = "egg"
t = "add"
```

The first character is different from the next two characters:

```text
egg -> 1 2 2
add -> 1 2 2
```

That shape matches, so the strings are isomorphic.

But:

```text
s = "foo"
t = "bar"
```

The shape is:

```text
foo -> 1 2 2
bar -> 1 2 3
```

The two `o` characters in `s` would need to become both `a` and `r`, which is impossible.

So the answer is `False`.

### 2. The Replacement Must Be a Bijection

The central word for this problem is **bijection**.

In this context, a bijection means:

```text
one source character maps to one target character
one target character is used by at most one source character
```

Both directions matter.

If we only check the direction from `s` to `t`, we catch this kind of conflict:

```text
s = "foo"
t = "bar"

f -> b
o -> a
o -> r   conflict: o was already mapped to a
```

But we also need to catch the opposite kind of conflict:

```text
s = "badc"
t = "baba"

b -> b
a -> a
d -> b   conflict: b is already used by source b
```

The mapping `d -> b` looks fine if we only ask whether `d` already had a mapping. It did not.

But it is invalid because target character `b` has already been claimed by source character `b`.

That is why the invariant must protect both directions.

### 3. Start From the Brute Force Baseline

A very direct way to think about the problem is:

1. Try to assign a replacement character for every distinct character in `s`.
2. Apply that replacement to all of `s`.
3. Check whether the result equals `t`.
4. Reject any assignment that maps two source characters to the same target character.

That is conceptually correct, but it is far too much work.

If there are many distinct characters, the number of possible assignments grows quickly. We would be exploring mappings that the input itself can disprove much earlier.

For example, in:

```text
s = "foo"
t = "bar"
```

we do not need to try every possible replacement table. The first two pairs give:

```text
f -> b
o -> a
```

The third pair immediately says:

```text
o -> r
```

That contradicts the existing mapping `o -> a`, so the answer is already impossible.

The repeated work in the brute-force idea is that it treats the mapping as something to guess. But the aligned character pairs in `s` and `t` already force the mapping.

### 4. Key Observation

At index `i`, the pair:

```text
s[i] must map to t[i]
```

is not optional.

If the strings are isomorphic, every aligned pair must agree with one single global replacement table.

So while scanning left to right, each position gives one local requirement:

```text
source character = s[i]
target character = t[i]
```

There are only three possibilities:

1. Neither character has been seen in a mapping yet.
2. They have been seen, and the old mapping agrees with this pair.
3. The pair contradicts the old mapping in at least one direction.

Only the third case makes the answer `False`.

This turns the problem into an invariant-maintenance problem.

### 5. The Bijection Invariant

Maintain two hash tables:

```text
forward[source] = target
backward[target] = source
```

The invariant after processing the first `i` positions is:

```text
For every processed position j < i:
forward[s[j]] == t[j]
backward[t[j]] == s[j]

and no two different source characters map to the same target character.
```

The two tables are mirror images of the same replacement rule.

They answer different questions:

```text
forward asks:  has this source character already chosen a target?
backward asks: has this target character already been claimed by a source?
```

Using only `forward` is incomplete because it allows two source characters to share one target.

Using only `backward` is incomplete because it allows one source character to point to multiple targets.

The bijection requires both.

### 6. Detailed Algorithm

First, if the strings have different lengths, return `False`.

In the LeetCode version, the input lengths are usually equal by construction, but the check is still the first-principles guard: a position-by-position replacement cannot turn a string of one length into a string of another length.

Then scan the strings together:

```text
for each aligned pair (source, target):
```

At every pair:

1. If `source` is already in `forward`, then it must map to this exact `target`.
2. If `target` is already in `backward`, then it must be claimed by this exact `source`.
3. If either check disagrees, return `False`.
4. Otherwise record both directions:

```text
forward[source] = target
backward[target] = source
```

If the scan finishes without a contradiction, return `True`.

The algorithm is not trying to build the transformed string. It is checking whether the forced replacement table remains internally consistent.

### 7. Walkthrough: `egg` and `add`

Let:

```text
s = "egg"
t = "add"
```

Start with empty tables:

```text
forward  = {}
backward = {}
```

#### Index 0: `e` with `a`

No mapping exists yet.

Record:

```text
forward[e] = a
backward[a] = e
```

Tables:

```text
forward  = {e: a}
backward = {a: e}
```

#### Index 1: `g` with `d`

No mapping exists yet.

Record:

```text
forward[g] = d
backward[d] = g
```

Tables:

```text
forward  = {e: a, g: d}
backward = {a: e, d: g}
```

#### Index 2: `g` with `d`

Now `g` is already mapped:

```text
forward[g] == d
```

And `d` is already claimed:

```text
backward[d] == g
```

Both checks agree with the current pair.

The scan ends with no contradiction, so:

```text
return True
```

### 8. Walkthrough: `foo` and `bar`

Let:

```text
s = "foo"
t = "bar"
```

#### Index 0: `f` with `b`

Record:

```text
f -> b
b -> f
```

#### Index 1: `o` with `a`

Record:

```text
o -> a
a -> o
```

#### Index 2: `o` with `r`

The source character `o` already has a required target:

```text
forward[o] == a
```

But the current aligned target is:

```text
r
```

That would require one source character to map to two different target characters:

```text
o -> a
o -> r
```

This violates the replacement rule, so:

```text
return False
```

### 9. Walkthrough: Why One Direction Is Not Enough

Consider:

```text
s = "badc"
t = "baba"
```

Scanning with only `forward` would produce:

```text
b -> b
a -> a
d -> b
c -> a
```

Every source character maps consistently to one target character, so a one-direction check would incorrectly accept it.

But the target character `b` is used twice:

```text
b in t is claimed by source b
b in t is also claimed by source d
```

That means two different source characters would collapse into the same target character.

The actual transformation would not be reversible as a character pattern, so the strings are not isomorphic.

The `backward` table catches this at index 2:

```text
current pair: d with b
backward[b] is already b, not d
```

Therefore the algorithm returns `False`.

### 10. Code

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        forward = {}
        backward = {}

        for source, target in zip(s, t):
            if source in forward and forward[source] != target:
                return False

            if target in backward and backward[target] != source:
                return False

            forward[source] = target
            backward[target] = source

        return True
```

The assignments at the end are safe even when the mapping already exists, because the contradiction checks have already proved that the existing value agrees with the current pair.

### 11. Equivalent Pattern-Encoding View

Another way to express the same idea is to convert each string into the pattern of first occurrences.

For example:

```text
"egg"   -> [0, 1, 1]
"add"   -> [0, 1, 1]
"paper" -> [0, 1, 0, 2, 3]
"title" -> [0, 1, 0, 2, 3]
```

Two strings are isomorphic exactly when these encoded patterns are equal.

Pseudocode:

```python
def encode(word):
    first_id = {}
    pattern = []

    for ch in word:
        if ch not in first_id:
            first_id[ch] = len(first_id)
        pattern.append(first_id[ch])

    return pattern

return encode(s) == encode(t)
```

This works because it records the equality structure directly. Characters that appeared for the first time at the same relative moments get the same ids, and repeated characters reuse their original ids.

The two-map bijection version is usually more direct for this problem because it checks the replacement rule exactly as stated.

### 12. Why the Algorithm Is Correct

We prove the two-map algorithm correct using the invariant.

#### Invariant

After processing any prefix of the strings, `forward` and `backward` describe a valid bijection for exactly the aligned pairs already processed.

That means:

```text
Every processed source character maps to the target character it has always appeared with.
Every processed target character is claimed by only the source character it has always appeared with.
```

#### Initialization

Before processing any characters, both tables are empty.

An empty mapping is a valid bijection for an empty prefix, so the invariant holds.

#### Maintenance

Suppose the invariant holds before processing a pair `(source, target)`.

If `source` already maps to a different target, then any full transformation would require `source` to have two outputs. That is impossible, so returning `False` is correct.

If `target` is already claimed by a different source, then any full transformation would require two source characters to share one output. That is also impossible, so returning `False` is correct.

If neither contradiction occurs, assigning:

```text
forward[source] = target
backward[target] = source
```

either adds a new consistent pair or repeats an already consistent pair. In both cases, the tables still describe a valid bijection for the processed prefix.

So the invariant is preserved.

#### Termination

If the algorithm returns `False`, it found a direct violation of the one-to-one replacement rule, so the strings cannot be isomorphic.

If the scan finishes, the invariant holds for the entire strings. Therefore every aligned pair agrees with one global bijection from characters of `s` to characters of `t`.

So the strings are isomorphic, and returning `True` is correct.

### 13. Complexity

Let `n` be the length of the strings.

The scan visits each aligned pair once.

Each hash table lookup and assignment is expected `O(1)`.

So:

```text
Time:  O(n)
Space: O(k)
```

where `k` is the number of distinct characters that appear in the strings.

In the worst case, `k` can be `O(n)`, so worst-case auxiliary space is `O(n)`.

### 14. Common Pitfalls

- Checking only `s -> t` and forgetting that two source characters cannot share one target character.
- Checking only character frequencies. `"foo"` and `"app"` have similar frequency shapes, but aligned positions still matter.
- Sorting the strings. Sorting destroys the position-by-position relationship that defines the replacement.
- Building the transformed string without enforcing that target characters are uniquely claimed.
- Updating the dictionaries before checking for conflicts, then accidentally overwriting the evidence of a contradiction.
- Treating different-length strings as possibly isomorphic. A character replacement preserves length.

### 15. First-Principles Summary

This problem follows from five basic facts:

```text
1. A character replacement preserves positions and length.
2. Each aligned pair s[i], t[i] forces one mapping requirement.
3. The same source character must always force the same target character.
4. The same target character cannot be forced by two different source characters.
5. Two hash tables store exactly those two directions of consistency.
```

So the whole algorithm is:

> Scan both strings together, maintain a forward and backward character mapping, reject the first contradiction, and accept only if every aligned pair is consistent with one bijection.

## Implementation

See `solutions/hash_table/p205_isomorphic_strings.py`.

## Tests

See `tests/hash_table/test_p205_isomorphic_strings.py`.

## Examples

### Example 1
- Input: `s = "egg"`, `t = "add"`
- Output: `true`
- Reason: `e -> a` and `g -> d` gives a consistent one-to-one mapping.

### Example 2
- Input: `s = "foo"`, `t = "bar"`
- Output: `false`
- Reason: `o` would need to map to both `a` and `r`.

### Example 3
- Input: `s = "paper"`, `t = "title"`
- Output: `true`
- Reason: `p -> t`, `a -> i`, `e -> l`, and `r -> e` are consistent, and the repeated `p` matches the repeated `t`.

## Follow-up Practice

- Trace `s = "badc"`, `t = "baba"` and identify which direction detects the failure.
- Explain why two maps are equivalent to enforcing a bijection.
- Write the pattern-encoding version and compare its invariant with the two-map version.
