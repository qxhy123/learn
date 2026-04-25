# 208. Implement Trie (Prefix Tree)

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/implement-trie-prefix-tree/
- Official Group: Trie
- Pattern Group: Trie
- Patterns: trie, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

We need to design a data structure called `Trie` that supports three operations:

```text
insert(word)
search(word)
startsWith(prefix)
```

The operations mean different things:

```text
insert("apple")
```

stores the whole word `"apple"`.

```text
search("apple")
```

returns `true` only if the exact word `"apple"` was inserted.

```text
startsWith("app")
```

returns `true` if at least one inserted word begins with `"app"`.

The important distinction is:

```text
A prefix existing is not the same thing as a word existing.
```

After inserting only:

```text
"apple"
```

we should have:

```text
search("apple")    -> true
search("app")      -> false
startsWith("app")  -> true
```

Why is `search("app")` false?

Because `"app"` is only a prefix of an inserted word. It was not inserted as a complete word.

So the data structure must remember two kinds of information:

1. Which character paths exist.
2. Which of those paths represent completed inserted words.

---

### 2. Start From the Brute Force Baseline

The simplest possible design is to store every inserted word in a list:

```python
words = []
```

Then:

```python
def insert(word):
    words.append(word)
```

Exact search is straightforward:

```python
def search(word):
    for stored_word in words:
        if stored_word == word:
            return True
    return False
```

Prefix search checks every stored word:

```python
def startsWith(prefix):
    for stored_word in words:
        if stored_word.startswith(prefix):
            return True
    return False
```

This works, but it repeats a lot of work.

Suppose we insert:

```text
apple
app
application
apply
apt
```

Every one of these words begins with `a`, and several begin with `app`.

A list stores them separately:

```text
apple
app
application
apply
apt
```

The shared characters are duplicated again and again.

For `startsWith("app")`, the list-based design may scan many full words even though the question only cares about whether the path:

```text
a -> p -> p
```

exists.

The brute-force design has the wrong shape for the question.

It treats each word as an independent object, but prefix questions are about shared beginnings.

---

### 3. The Key Observation: Prefixes Form a Tree

If many words share prefixes, we should store each shared prefix once.

For example, the words:

```text
app
apple
apply
apt
```

can be represented as paths from a common root:

```text
(root)
  |
  a
  |
  p
  |\
  p t
  |
  l
 / \
e   y
```

Each edge represents one character.

A path from the root spells a prefix:

```text
root -> a                 spells "a"
root -> a -> p            spells "ap"
root -> a -> p -> p       spells "app"
root -> a -> p -> p -> l  spells "appl"
```

A trie is exactly this idea:

> Store strings by sharing their common prefixes in a tree of characters.

The root represents the empty prefix:

```text
""
```

The root's child `a` represents prefix:

```text
"a"
```

The node reached by `a -> p -> p` represents prefix:

```text
"app"
```

Now prefix lookup becomes natural:

```text
Can I follow the characters of the prefix from the root?
```

If yes, some inserted word has that prefix.

Exact word lookup needs one extra fact:

```text
Was this node marked as the end of an inserted word?
```

---

### 4. Trie Node Invariant

Each trie node represents one prefix.

We do not need to store the full prefix inside every node. The prefix is implied by the path taken from the root to that node.

For every node, maintain:

```text
children    = mapping from next character to next node
is_word_end = whether this node finishes an inserted word
```

The core invariant is:

```text
For any node reached by following characters c1, c2, ..., ck from the root,
that node represents exactly the prefix c1c2...ck.
```

A child edge extends the prefix by one character:

```text
current node represents "app"
child edge 'l'
child node represents "appl"
```

The `is_word_end` marker answers whether this prefix is also a complete inserted word.

For example, after inserting only `"apple"`:

```text
(root)
  |
  a
  |
  p
  |
  p
  |
  l
  |
  e*      * means is_word_end = true
```

The node for `"app"` exists, but it is not marked as a word ending.

So:

```text
startsWith("app") -> true
search("app")     -> false
```

After inserting `"app"` too:

```text
(root)
  |
  a
  |
  p
  |
  p*      now "app" is a complete word
  |
  l
  |
  e*
```

Now:

```text
startsWith("app") -> true
search("app")     -> true
```

This invariant is the entire data structure.

Every operation is just a way of preserving or querying it.

---

### 5. Detailed Algorithm

#### Initialization

Create one root node:

```text
root.children = {}
root.is_word_end = false
```

The root does not represent a real character. It represents the empty prefix.

---

#### `insert(word)`

To insert a word, start at the root and process characters left to right.

For each character:

1. Check whether the current node already has a child for that character.
2. If not, create a new child node.
3. Move to that child.

After the last character, mark the current node as a word ending.

Pseudocode:

```python
def insert(word):
    node = root

    for ch in word:
        if ch not in node.children:
            node.children[ch] = TrieNode()
        node = node.children[ch]

    node.is_word_end = True
```

Why mark only after the loop?

Because only the full path spells the full word.

For `"apple"`, the nodes for `"a"`, `"ap"`, `"app"`, and `"appl"` are prefixes, but the node for `"apple"` is the completed word.

---

#### `search(word)`

To search for an exact word, start at the root and follow each character.

If at any point the needed child does not exist, the word cannot have been inserted.

If all characters can be followed, we still are not done. We must check whether the final node is marked as a complete word.

Pseudocode:

```python
def search(word):
    node = root

    for ch in word:
        if ch not in node.children:
            return False
        node = node.children[ch]

    return node.is_word_end
```

The final line is what separates exact word search from prefix search.

---

#### `startsWith(prefix)`

Prefix search also starts at the root and follows characters.

If any character edge is missing, no inserted word has this prefix.

If every character can be followed, the prefix exists.

We do not care whether the final node is a complete word.

Pseudocode:

```python
def startsWith(prefix):
    node = root

    for ch in prefix:
        if ch not in node.children:
            return False
        node = node.children[ch]

    return True
```

This is why `startsWith("app")` can return `true` even when `search("app")` returns `false`.

---

### 6. A Helpful Shared Helper

Both `search` and `startsWith` need to walk a string from the root.

A clean implementation often uses a helper:

```python
def _find_node(text):
    node = root

    for ch in text:
        if ch not in node.children:
            return None
        node = node.children[ch]

    return node
```

Then:

```python
def search(word):
    node = _find_node(word)
    return node is not None and node.is_word_end
```

and:

```python
def startsWith(prefix):
    return _find_node(prefix) is not None
```

This avoids duplicating traversal logic.

The helper's meaning is precise:

```text
Return the node representing this string if its path exists;
otherwise return None.
```

---

### 7. Detailed Example Walkthrough

Use the official operation sequence:

```text
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[],     ["apple"], ["apple"], ["app"], ["app"],      ["app"], ["app"]]
```

#### Step 1: Create the trie

Start with only the root:

```text
(root)
```

No words exist yet.

---

#### Step 2: `insert("apple")`

Start at the root.

Character `a` is missing, so create it:

```text
(root)
  |
  a
```

Character `p` is missing under `a`, so create it:

```text
(root)
  |
  a
  |
  p
```

Next character `p` is missing under the first `p`, so create it:

```text
(root)
  |
  a
  |
  p
  |
  p
```

Character `l` is missing, so create it:

```text
(root)
  |
  a
  |
  p
  |
  p
  |
  l
```

Character `e` is missing, so create it and mark it as a word ending:

```text
(root)
  |
  a
  |
  p
  |
  p
  |
  l
  |
  e*
```

Now the trie contains exactly one complete word:

```text
apple
```

---

#### Step 3: `search("apple")`

Follow the path:

```text
a -> p -> p -> l -> e
```

Every edge exists.

The final node `e` is marked as a word ending.

So:

```text
search("apple") -> true
```

---

#### Step 4: `search("app")`

Follow the path:

```text
a -> p -> p
```

Every edge exists.

But the final node for `"app"` is not marked as a word ending.

It is only an internal prefix node on the way to `"apple"`.

So:

```text
search("app") -> false
```

---

#### Step 5: `startsWith("app")`

Again follow:

```text
a -> p -> p
```

Every edge exists.

For prefix search, this is enough. We do not require the final node to be marked as a word ending.

So:

```text
startsWith("app") -> true
```

---

#### Step 6: `insert("app")`

Start at the root and follow existing edges:

```text
a -> p -> p
```

No new nodes are needed because `"app"` is already a prefix of `"apple"`.

Now mark the node for `"app"` as a word ending:

```text
(root)
  |
  a
  |
  p
  |
  p*      marks "app"
  |
  l
  |
  e*      marks "apple"
```

The trie now contains two complete words:

```text
app
apple
```

---

#### Step 7: `search("app")`

Follow:

```text
a -> p -> p
```

The path exists, and now the final node is marked as a word ending.

So:

```text
search("app") -> true
```

This example shows the central idea of the problem:

```text
A node can exist because it is a prefix,
but search only succeeds when that node is marked as a complete word.
```

---

### 8. Reference Implementation Shape

A Python implementation can represent each node with a dictionary and a Boolean marker:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False
```

Then the trie owns one root node:

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()
```

Full implementation:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root

        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        node.is_word_end = True

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_word_end

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def _find_node(self, text: str):
        node = self.root

        for ch in text:
            if ch not in node.children:
                return None
            node = node.children[ch]

        return node
```

Some implementations use fixed-size arrays of length `26` instead of dictionaries because the constraints say words contain lowercase English letters.

Both designs express the same invariant:

```text
From a node, each outgoing character leads to the node for the prefix extended by that character.
```

A dictionary is usually easier to read and avoids allocating unused child slots.

An array can be faster and more memory-predictable when the alphabet is fixed.

---

### 9. Correctness

We prove that the trie operations return the required results.

#### Lemma 1: After inserting a word, the trie contains a path spelling that word.

During `insert(word)`, the algorithm starts at the root and processes characters in order.

For each character, if the required child edge already exists, the algorithm follows it. If it does not exist, the algorithm creates it and then follows it.

Therefore, after processing the first `k` characters, the current node represents the prefix `word[0:k]`.

After all characters are processed, the trie contains the full path for `word`.

#### Lemma 2: After inserting a word, the final node of that word is marked as a complete word.

At the end of `insert(word)`, the current node is exactly the node reached by following all characters of `word` from the root.

The algorithm sets:

```text
is_word_end = true
```

on that node.

Therefore, the node representing `word` is marked as a complete inserted word.

#### Lemma 3: `search(word)` returns `true` only if `word` was inserted.

`search(word)` first follows the characters of `word` from the root.

If any edge is missing, there is no path spelling `word`, so `word` could not have been inserted.

If the path exists, `search` returns `true` only when the final node has `is_word_end = true`.

The only operation that sets `is_word_end` on the node for a string is inserting that exact string.

Therefore, `search(word)` returns `true` only if `word` was inserted.

#### Lemma 4: If `word` was inserted, `search(word)` returns `true`.

By Lemma 1, insertion creates or preserves the full path for `word`.

By Lemma 2, insertion marks the final node of that path as a complete word.

So when `search(word)` follows the same characters, it reaches that final node and sees `is_word_end = true`.

Therefore, `search(word)` returns `true`.

#### Lemma 5: `startsWith(prefix)` returns `true` exactly when some inserted word has that prefix.

If `startsWith(prefix)` returns `true`, then the path for `prefix` exists from the root.

Such a path can only be created by inserting a word whose first characters include that path, so at least one inserted word has that prefix.

Conversely, if some inserted word has `prefix` as its prefix, then inserting that word created or reused every edge along the path for `prefix`.

So `startsWith(prefix)` can follow all characters of `prefix` and returns `true`.

#### Theorem: The trie implementation is correct.

From Lemmas 3 and 4, `search` returns `true` exactly for inserted words.

From Lemma 5, `startsWith` returns `true` exactly for prefixes of inserted words.

`insert` establishes the paths and word-ending markers required by those operations.

Therefore, all required operations behave correctly.

---

### 10. Complexity

Let:

```text
L = length of the input word or prefix for one operation
N = total number of characters inserted across all words
```

#### `insert(word)`

The algorithm processes each character once.

```text
Time:  O(L)
Space: O(L) in the worst case for newly created nodes
```

If the word shares existing prefixes, fewer than `L` nodes may be created.

#### `search(word)`

The algorithm follows at most one edge per character.

```text
Time:  O(L)
Space: O(1) extra, ignoring the stored trie
```

#### `startsWith(prefix)`

The algorithm also follows at most one edge per character.

```text
Time:  O(L)
Space: O(1) extra, ignoring the stored trie
```

#### Total Stored Trie Size

Each newly created node corresponds to one distinct prefix created by some insertion.

In the worst case, no words share prefixes.

Then the number of nodes is proportional to the total number of inserted characters:

```text
Space: O(N)
```

There is also one root node.

---

### 11. Common Pitfalls

#### Pitfall 1: Treating prefix existence as word existence

This is the most common bug.

If `"apple"` was inserted, the path for `"app"` exists.

But:

```text
search("app")
```

must return `false` unless `"app"` was inserted separately.

That is why `search` must check `is_word_end`.

---

#### Pitfall 2: Forgetting to mark the final node during insertion

Creating the path is not enough.

After inserting `"apple"`, the final node must be marked:

```text
e.is_word_end = true
```

Otherwise `search("apple")` would incorrectly return `false`.

---

#### Pitfall 3: Marking every prefix as a word

While inserting `"apple"`, do not mark `"a"`, `"ap"`, `"app"`, and `"appl"` as word endings.

Only the final node for `"apple"` should be marked.

Otherwise `search("app")` would incorrectly return `true` after inserting only `"apple"`.

---

#### Pitfall 4: Starting traversal from the wrong node

Every operation starts from the root.

The root is the common starting point for all words.

If traversal state is accidentally reused between operations, later operations may start in the middle of the trie and return incorrect answers.

---

#### Pitfall 5: Confusing missing child with failed word ending

There are two different failure cases for `search`:

```text
The path does not exist.
The path exists, but the final node is not a word ending.
```

For `startsWith`, only the first case is a failure.

Keeping these cases separate makes the code much easier to reason about.

---

#### Pitfall 6: Overcomplicating with recursion

This problem does not require recursion.

Each operation follows one simple path from the root.

A loop over the characters is enough.

Recursion becomes useful in related trie problems with wildcard matching or board search, but not here.

---

### 12. First-Principles Summary

The problem asks for a data structure that can answer exact-word and prefix queries after insertions.

A list of words is correct but inefficient for prefix queries because it stores shared prefixes repeatedly and may scan many words.

The first-principles shift is:

```text
Words are sequences of characters.
Prefixes are shared beginnings of those sequences.
Shared beginnings should be represented once.
```

A trie does this by storing one character per edge and one prefix per node.

The root represents the empty prefix.

Following edges from the root spells a prefix.

The node invariant is:

```text
The path from the root to a node spells exactly the prefix represented by that node.
```

The word-ending marker adds the missing distinction:

```text
This prefix exists as a path.
This prefix was inserted as a complete word.
```

Then the three operations become direct consequences of the invariant:

```text
insert      = create/follow the path, then mark the final node
search      = follow the path, then require the final node to be marked
startsWith  = follow the path, and only require the path to exist
```

That is the whole problem.

The trie is not an arbitrary tree trick. It is the natural shape of the information the operations ask for.

## Implementation
See `solutions/trie/p208_implement_trie_prefix_tree.py`.

## Tests
See `tests/trie/test_p208_implement_trie_prefix_tree.py`.

## Examples

### Example 1
- Input: `{'raw': '["Trie","insert","search","search","startsWith","insert","search"]\n[[],["apple"],["apple"],["app"],["app"],["app"],["app"]]'}`
- Output: `'See official examples'`

## Follow-up Practice
- Draw the trie after inserting `"apple"`, then mark exactly which node makes `search("apple")` true.
- Explain why `startsWith("app")` is true before `search("app")` becomes true.
- Insert `"app"` after `"apple"` and identify which nodes are reused versus newly created.
- Implement the traversal helper once, then use it for both `search` and `startsWith`.
