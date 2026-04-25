# 211. Design Add and Search Words Data Structure

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/design-add-and-search-words-data-structure/
- Official Group: Trie
- Pattern Group: Trie
- Patterns: trie

## First-Principles Explanation

### What The Problem Is Asking
The problem asks us to design a mutable word dictionary with two operations:

- `addWord(word)`: store `word` in the dictionary.
- `search(word)`: return whether the dictionary contains any stored word that matches `word`.

The twist is that the search pattern may contain the character `.`. A dot does not mean a literal period; it means "match exactly one arbitrary lowercase letter." So:

- `bad` matches only the exact word `bad`.
- `.ad` matches `bad`, `dad`, `mad`, and any other three-letter word whose last two letters are `ad`.
- `b..` matches any three-letter word starting with `b`.
- `b.` does not match `bad`, because each `.` consumes exactly one character, not zero or many.

This is not a prefix-existence problem. If `bad` was inserted, then `ba` is a prefix in the internal structure, but `search("ba")` must still be `False` unless `ba` was also inserted as a full word.

The data structure must therefore remember two facts:

1. Which character paths exist among inserted words.
2. Which of those paths represent complete inserted words.

### Brute-Force Baseline
The simplest possible implementation is to keep all inserted words in a list or set.

For `addWord(word)`, append it to the collection.

For `search(pattern)`, compare the pattern against every stored word:

1. Skip any candidate whose length is different from the pattern length.
2. For each character position:
   - If the pattern character is a normal letter, it must equal the candidate character.
   - If the pattern character is `.`, it matches the candidate character automatically.
3. If any candidate passes every position, return `True`.
4. Otherwise return `False`.

That baseline is easy to reason about, but it repeats the same work across words. If the dictionary contains many words beginning with the same prefix, every search re-checks that prefix again and again. For example, searching `app..` in a list containing `apple`, `apply`, `aptly`, `angle`, and `arena` repeatedly inspects early characters even though many candidates can be rejected after a shared prefix decision.

If there are `N` stored words and the pattern length is `L`, brute-force search costs `O(N * L)` in the worst case. `addWord` can be cheap, but searches become expensive as the dictionary grows.

### Key Observation
Words are not independent sequences when they share prefixes. If several inserted words begin with the same characters, those common characters can be represented once.

A trie does exactly that:

```text
root
 ├── b
 │   └── a
 │       └── d  (word ends here: bad)
 ├── d
 │   └── a
 │       └── d  (word ends here: dad)
 └── m
     └── a
         └── d  (word ends here: mad)
```

Each edge corresponds to one character. Each root-to-node path spells one prefix. A node's `is_word` marker says whether that prefix is a full inserted word.

Exact-letter search is then deterministic: from the current node, follow the edge with that character. If the edge does not exist, no stored word can match.

Wildcard search is the only new ingredient. When the pattern character is `.`, we do not know which edge to follow. But we do know something very precise: the dot must consume exactly one character, so it may follow any one child edge of the current trie node, then continue matching the next pattern position.

That turns the problem into a controlled tree search over only prefixes that actually exist in the dictionary.

### Trie And Wildcard Invariant
During search, maintain this invariant:

> After consuming the first `i` characters of the search pattern, every active recursive call is positioned at a trie node whose root-to-node path matches those `i` pattern characters, respecting `.` as exactly one arbitrary character.

This invariant is the whole algorithm.

- At `i = 0`, no pattern characters have been consumed, and the only active node is the root. The empty path matches the empty prefix.
- If `pattern[i]` is a letter, the next active node can only be the child reached by that letter.
- If `pattern[i]` is `.`, the next active nodes are all children of the current node, because each child represents choosing one real character for the dot.
- When `i == len(pattern)`, all pattern characters have been consumed. The search succeeds only if the current node is marked as the end of an inserted word.

The end marker is essential. Reaching some trie node after consuming the whole pattern means the pattern matches a stored prefix. It is a successful word match only when that prefix was explicitly inserted as a word.

### Detailed Algorithm
Use a trie node with:

- `children`: a mapping from character to child trie node.
- `is_word`: a boolean indicating whether a word ends at this node.

#### `addWord(word)`

1. Start at the root.
2. For each character `ch` in `word`:
   - If the current node has no child for `ch`, create one.
   - Move to that child.
3. After the last character, set `is_word = True` on the current node.

This inserts every prefix of `word` as a path, and marks only the complete word as searchable.

#### `search(pattern)`

Run a depth-first helper `dfs(node, index)` meaning:

> Can the suffix `pattern[index:]` match some word suffix starting from this trie node?

The helper works as follows:

1. If `index == len(pattern)`, return `node.is_word`.
2. Let `ch = pattern[index]`.
3. If `ch` is a normal letter:
   - If `ch` is not in `node.children`, return `False`.
   - Otherwise return `dfs(node.children[ch], index + 1)`.
4. If `ch == '.'`:
   - Try every child of `node`.
   - If any child returns `True` for `dfs(child, index + 1)`, return `True` immediately.
   - If no child works, return `False`.

The early return on wildcard branches is safe because the question asks whether at least one stored word matches, not how many words match.

### Pseudocode

```text
class TrieNode:
    children = map from char to TrieNode
    is_word = false

class WordDictionary:
    root = TrieNode()

    addWord(word):
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = true

    search(pattern):
        return dfs(root, 0)

    dfs(node, index):
        if index == length(pattern):
            return node.is_word

        ch = pattern[index]

        if ch != '.':
            if ch not in node.children:
                return false
            return dfs(node.children[ch], index + 1)

        for child in node.children.values():
            if dfs(child, index + 1):
                return true
        return false
```

### Python-Style Implementation Sketch

```python
class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word = False


class WordDictionary:
    def __init__(self) -> None:
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, index: int) -> bool:
            if index == len(word):
                return node.is_word

            ch = word[index]
            if ch != ".":
                child = node.children.get(ch)
                if child is None:
                    return False
                return dfs(child, index + 1)

            return any(dfs(child, index + 1) for child in node.children.values())

        return dfs(self.root, 0)
```

The repository's solution file is currently scaffolded, but this is the intended data-structure shape for the LeetCode interface.

### Detailed Example Walkthrough

Use the official operation sequence:

```text
WordDictionary()
addWord("bad")
addWord("dad")
addWord("mad")
search("pad")
search("bad")
search(".ad")
search("b..")
```

After the three insertions, the trie contains these complete word paths:

```text
root
 ├── b -> a -> d  [word]
 ├── d -> a -> d  [word]
 └── m -> a -> d  [word]
```

#### Search `"pad"`

Start at the root with index `0`.

1. Pattern character `p` is a normal letter.
2. The root has children `b`, `d`, and `m`, but no `p` child.
3. No inserted word begins with `p`, so the search returns `False` immediately.

#### Search `"bad"`

1. At the root, `b` exists, so move to the `b` node.
2. At index `1`, `a` exists under `b`, so move to the `ba` node.
3. At index `2`, `d` exists under `ba`, so move to the `bad` node.
4. The pattern is consumed. The `bad` node has `is_word = True`, so return `True`.

#### Search `".ad"`

1. At the root, the first pattern character is `.`.
2. The dot can match exactly one child edge: try `b`, `d`, or `m`.
3. Suppose the search tries `b` first. The remaining pattern is `ad`.
4. From the `b` node, follow `a` to `ba`.
5. From `ba`, follow `d` to `bad`.
6. The pattern is consumed and `bad` is a word, so return `True`.

The search does not need to inspect the `d` and `m` branches after finding one successful branch.

#### Search `"b.."`

1. At the root, follow exact letter `b`.
2. At the `b` node, the next pattern character is `.`. The only child is `a`, so the dot matches `a`.
3. At the `ba` node, the final pattern character is another `.`. The only child is `d`, so the dot matches `d`.
4. The pattern is consumed at the `bad` node, and `is_word = True`, so return `True`.

Notice that `b..` matches `bad` because both dots consume one character each. It would not match a two-letter word beginning with `b`, and it would not match a four-letter word beginning with `b`.

### Correctness

We prove that `search(pattern)` returns `True` if and only if some inserted word matches `pattern`.

#### Lemma 1: Inserted words are represented exactly by terminal trie nodes

When `addWord(word)` runs, it creates or reuses one edge for each character of `word`, starting from the root, then marks the final node as `is_word`. Therefore the root-to-final-node path spells `word`, and that node is terminal. The algorithm never marks intermediate prefix nodes unless those prefixes are inserted separately. So terminal nodes correspond exactly to inserted words.

#### Lemma 2: Each recursive search state represents exactly the pattern prefix consumed so far

Consider a call `dfs(node, index)`. By construction, the path from the root to `node` matches `pattern[:index]`.

- Initially, `dfs(root, 0)` has consumed no characters, so the empty trie path matches the empty pattern prefix.
- For a normal letter, the algorithm recurses only to the child with that exact letter, preserving the match for one additional pattern character.
- For `.`, the algorithm recurses to each child. Each child edge contributes exactly one real character, which is exactly what `.` is allowed to match.

Thus every recursive call preserves the invariant.

#### Lemma 3: The recursive search explores every possible matching trie path

At each normal letter, there is only one possible next edge that could match, and the algorithm follows it if it exists. At each `.`, every child edge is a possible one-character match, and the algorithm tries all children. Therefore no trie path that could match the pattern is skipped.

#### Theorem: `search(pattern)` is correct

If the algorithm returns `True`, it reached `index == len(pattern)` at a node with `is_word = True`. By Lemma 2, that node's path matches the entire pattern. By Lemma 1, that path is an inserted word. So a matching inserted word exists.

If a matching inserted word exists, its trie path is present and terminal by Lemma 1. By Lemma 3, the recursive search follows the exact required edge at each normal letter and includes the word's edge among the branches at each dot. It therefore reaches that terminal node after consuming the whole pattern and returns `True`.

So `search(pattern)` returns `True` exactly when the dictionary contains a word matching the pattern.

### Complexity

Let `L` be the length of the word or search pattern, and let `Σ` be the alphabet size. For lowercase English letters, `Σ = 26`.

#### `addWord(word)`

- Time: `O(L)`, because it processes each character once.
- Space: `O(L)` additional space in the worst case, when none of the word's prefixes already exist in the trie. Across all inserted words, total trie space is `O(total inserted characters)` in the worst case.

#### `search(pattern)`

- Best/typical exact search time: `O(L)` when there are no wildcards, because each character determines at most one next node.
- Worst-case wildcard search time: `O(Σ^L)` as a loose upper bound when every pattern character is `.` and the trie is dense enough to branch at every level.
- Tighter practical bound: `O(number of trie nodes visited)`, which is never more than the number of nodes at depths `0` through `L`.
- Recursion stack: `O(L)`, because each recursive path consumes one pattern character per level.

The trie does not magically remove wildcard branching. Its value is that it branches only through prefixes that actually exist among inserted words, instead of comparing against every full stored word from scratch.

### Common Pitfalls

- Forgetting `is_word`: A path existing in the trie means a prefix exists, not necessarily a complete word.
- Treating `.` like `*`: In this problem, `.` matches exactly one character. It does not match zero characters or an arbitrary-length substring.
- Returning `False` too early for wildcard search: If one child branch fails, another child branch may still succeed.
- Returning `True` before consuming the whole pattern: A matching prefix is not enough; the pattern length and word length must both be fully matched.
- Using one mutable default dictionary for all nodes: Each trie node needs its own `children` mapping.
- Forgetting repeated insertions are harmless: Adding the same word again should leave the terminal marker `True`.
- Confusing the LeetCode class name: The platform uses `WordDictionary`, even if a local scaffold may expose a placeholder `Solution` class.

### First-Principles Summary

The dictionary needs to answer whether a pattern describes at least one inserted word. A list stores words independently, so every search repeats character comparisons across all candidates. A trie stores shared prefixes once, turning exact letters into deterministic edge choices.

The wildcard `.` does not break the trie model; it simply changes one deterministic edge choice into a finite set of possible child choices. The invariant is: after consuming `i` pattern characters, the current trie node represents a prefix that matches those `i` characters. Exact letters preserve the invariant by following one named child; dots preserve it by trying every child; the base case checks `is_word` to distinguish complete words from prefixes.

From first principles, the algorithm is just this invariant translated into code: insert by creating character paths, search by recursively matching the pattern against existing trie paths, and accept only when the entire pattern ends at an inserted word.

## Implementation
See `solutions/trie/p211_design_add_and_search_words_data_structure.py`.

## Tests
See `tests/trie/test_p211_design_add_and_search_words_data_structure.py`.

## Examples

### Example 1
- Input: `{'raw': '["WordDictionary","addWord","addWord","addWord","search","search","search","search"]\n[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]'}'
- Output: `'See official examples'`

Expanded operation results:

```text
WordDictionary() -> null
addWord("bad") -> null
addWord("dad") -> null
addWord("mad") -> null
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
```

## Follow-up Practice
- Insert `a`, `at`, and `ate`, then trace why `search("a")`, `search("a.")`, and `search("a..")` ask three different length questions.
- Draw the recursion tree for `search("...")` after inserting `bad`, `dad`, and `mad`.
- Compare the number of visited trie nodes for `search("b..")` versus brute-force checking every stored word.
- Implement the same search iteratively with a stack of `(node, index)` states.
