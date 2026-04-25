# 212. Word Search II

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/word-search-ii/
- Official Group: Trie
- Pattern Group: Trie
- Patterns: trie

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
board = an m x n grid of lowercase letters
words = a list of candidate words
```

A word exists on the board if you can spell it by walking through adjacent cells.

Adjacent means four-directional movement only:

```text
up, down, left, right
```

You may not reuse the same board cell inside one word path.

For example:

```text
board = [
  ["o", "a", "a", "n"],
  ["e", "t", "a", "e"],
  ["i", "h", "k", "r"],
  ["i", "f", "l", "v"]
]
words = ["oath", "pea", "eat", "rain"]
```

The word `"oath"` exists:

```text
o -> a -> t -> h
(0,0) -> (0,1) -> (1,1) -> (2,1)
```

The word `"eat"` also exists:

```text
e -> a -> t
(1,3) -> (1,2) -> (1,1)
```

The word `"pea"` does not exist because there is no `p` on the board.

The word `"rain"` does not exist because the board cannot form that sequence by legal adjacent moves.

So the answer is:

```text
["eat", "oath"]
```

The output order usually does not matter on LeetCode, but this repository's scaffold examples show the expected list as written in the test data.

The real problem is:

> Among many candidate words, find exactly the words that can be traced as simple paths in the grid.

The hard part is not checking one word. The hard part is avoiding the repeated work of checking many words that share prefixes.

---

### 2. Start From the Brute Force Baseline

The simplest mental model is to check every word independently.

For each word:

1. Try every board cell as the starting position.
2. If the cell matches the first character, run DFS from that cell.
3. At DFS depth `i`, require the current cell to equal `word[i]`.
4. Move to a neighboring cell for `word[i + 1]`.
5. Mark cells as visited so the same cell is not reused in the same path.

Pseudocode:

```python
def exists(board, word):
    for row in range(rows):
        for col in range(cols):
            if dfs(row, col, 0, word):
                return True
    return False

answer = []
for word in words:
    if exists(board, word):
        answer.append(word)
```

This is correct because it directly tests the definition of a valid word path.

But it repeats a lot of work.

Suppose the word list contains:

```text
oath
oat
oats
oak
```

Checking them independently means the board paths for prefix `"oa"` are rediscovered again and again.

The worst-case branching is large. From a cell, the first step can branch to up to 4 neighbors; after that, each step can branch to up to 3 unvisited neighbors because returning to the previous cell is usually forbidden.

For one word of length `L`, a rough upper bound is:

```text
O(m * n * 4 * 3^(L - 1))
```

For `W` words, doing that separately can become:

```text
O(W * m * n * 4 * 3^(L - 1))
```

That baseline teaches the key question:

> Can we search the board once per shared prefix instead of once per word?

Yes. That is exactly what a trie gives us.

---

### 3. The Key Observation: Board Paths Grow One Character at a Time

A DFS path on the board spells a prefix.

For example, after visiting:

```text
(0,0) -> (0,1) -> (1,1)
```

on the sample board, the spelled string is:

```text
oat
```

At that moment, there are only two useful possibilities:

1. `"oat"` is not a prefix of any target word.
2. `"oat"` is a prefix of at least one target word.

If `"oat"` is not a prefix of any word, continuing the path is pointless.

Why?

Because every longer path starting with `"oat"` will still start with `"oat"`. If no target word has that prefix, no extension can ever become a target word.

This is the central pruning idea:

> During DFS, abandon a board path as soon as its spelled prefix is not a prefix of any candidate word.

A trie is the data structure that answers this prefix question while we walk character by character.

---

### 4. What the Trie Represents

Build a trie from all words.

Each trie edge is one character. Each root-to-node path spells a prefix of at least one word.

For:

```text
words = ["oath", "pea", "eat", "rain"]
```

part of the trie looks like:

```text
root
├── o
│   └── a
│       └── t
│           └── h  (word = "oath")
├── p
│   └── e
│       └── a      (word = "pea")
├── e
│   └── a
│       └── t      (word = "eat")
└── r
    └── a
        └── i
            └── n  (word = "rain")
```

The trie root means:

```text
no characters have been chosen yet
```

A trie node after following characters `"oa"` means:

```text
the current board path spells "oa", and at least one target word starts with "oa"
```

A terminal marker means:

```text
the current prefix is also a complete word to report
```

Many implementations store the full word at the terminal node instead of storing only a boolean. That avoids reconstructing the string from the DFS path.

For example:

```python
node.word = "oath"
```

at the terminal node for `"oath"`.

---

### 5. The Trie + DFS Invariant

The algorithm combines two states:

```text
board position: (row, col)
trie node: node
```

The most important invariant is:

> When DFS is about to continue from trie node `node`, the board cells already visited spell exactly the trie path from the root to `node`.

That means the DFS does not need to store and repeatedly compare a string prefix.

The trie node itself tells us:

```text
which prefixes are still possible from here
```

At a board cell with character `ch`, there is only one meaningful question:

```text
Does the current trie node have a child for ch?
```

If not, the board path plus this cell is not a prefix of any target word, so stop immediately.

If yes, move to that child node. Now the invariant remains true:

```text
visited board path == trie path to child
```

If the child node stores a complete word, report that word.

Then recursively try the four neighboring board cells, while temporarily marking the current cell as visited.

---

### 6. Detailed Algorithm

#### Step 1: Build the Trie

Create a root node.

For each word:

1. Start at the root.
2. For each character in the word:
   - create the child node if missing;
   - move to that child.
3. Store the complete word at the final node.

Conceptually:

```python
root = TrieNode()

for word in words:
    node = root
    for ch in word:
        if ch not in node.children:
            node.children[ch] = TrieNode()
        node = node.children[ch]
    node.word = word
```

The trie now represents every prefix that is worth exploring on the board.

#### Step 2: Start DFS From Every Board Cell

A valid word can start anywhere, so every cell is a possible DFS starting point.

```python
for row in range(rows):
    for col in range(cols):
        dfs(row, col, root)
```

The DFS receives the trie node for the prefix before consuming the current board cell.

#### Step 3: Consume the Current Cell

Inside DFS:

1. Read `ch = board[row][col]`.
2. If `ch` is not a child of the current trie node, return.
3. Move to `next_node = node.children[ch]`.

This is where prefix pruning happens.

#### Step 4: Report a Complete Word

If `next_node.word` is not empty, the current board path spells a target word.

Add it to the answer.

Then clear `next_node.word` so the same word is not reported again from another path.

```python
if next_node.word is not None:
    answer.append(next_node.word)
    next_node.word = None
```

This is a common duplicate-avoidance trick.

It is safe because the problem asks for each word at most once, not for every path that can spell it.

#### Step 5: Mark the Cell as Visited

A cell cannot be reused in the same word path.

Common approaches:

- maintain a separate `visited` set;
- temporarily overwrite the board cell with a sentinel such as `"#"`.

The in-place sentinel approach is concise:

```python
board[row][col] = "#"
```

Before returning, restore it:

```python
board[row][col] = ch
```

This backtracking restore is essential. The cell is forbidden only for the current path, not for all future starting paths.

#### Step 6: Explore Neighbors

Try the four legal directions:

```text
(row + 1, col)
(row - 1, col)
(row, col + 1)
(row, col - 1)
```

For each neighbor:

1. Check bounds.
2. Check it is not marked visited.
3. Recurse with `next_node`.

Optional optimization: after finishing a child node, remove it from its parent if it has no children and no word. This prunes dead trie branches after all words below that branch have already been found.

The core solution does not depend on that optimization, but it can improve runtime on dense boards and repeated prefixes.

---

### 7. Pseudocode

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None


def findWords(board, words):
    root = TrieNode()

    for word in words:
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.word = word

    rows = len(board)
    cols = len(board[0])
    answer = []

    def dfs(row, col, node):
        ch = board[row][col]

        if ch not in node.children:
            return

        next_node = node.children[ch]

        if next_node.word is not None:
            answer.append(next_node.word)
            next_node.word = None

        board[row][col] = "#"

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr = row + dr
            nc = col + dc

            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, next_node)

        board[row][col] = ch

        # Optional pruning:
        # if not next_node.children and next_node.word is None:
        #     del node.children[ch]

    for row in range(rows):
        for col in range(cols):
            dfs(row, col, root)

    return answer
```

The solution file in this repository is currently a scaffold, but an implementation following this tutorial would use this structure.

---

### 8. Walk Through the Sample Board

Use:

```text
board = [
  ["o", "a", "a", "n"],
  ["e", "t", "a", "e"],
  ["i", "h", "k", "r"],
  ["i", "f", "l", "v"]
]
words = ["oath", "pea", "eat", "rain"]
```

The trie root has children:

```text
o, p, e, r
```

#### Starting at `(0,0)` = `"o"`

`"o"` is a root child, so DFS continues.

Current state:

```text
board path: o
trie prefix: o
```

Neighbors include `(0,1)` = `"a"` and `(1,0)` = `"e"`.

From trie prefix `"o"`, the only child is `"a"`.

So moving to `(1,0)` = `"e"` is immediately rejected:

```text
oe is not a prefix of any target word
```

Moving to `(0,1)` = `"a"` continues:

```text
board path: o -> a
trie prefix: oa
```

From `"oa"`, the next needed character is `"t"`.

A legal neighbor `(1,1)` contains `"t"`, so DFS continues:

```text
board path: o -> a -> t
trie prefix: oat
```

From `"oat"`, the next needed character is `"h"`.

A legal neighbor `(2,1)` contains `"h"`, so DFS continues:

```text
board path: o -> a -> t -> h
trie prefix: oath
```

The trie node for `"oath"` stores the word `"oath"`, so append it to the answer.

Then clear that terminal word marker so another path cannot append `"oath"` a second time.

#### Starting at `(1,3)` = `"e"`

`"e"` is a root child, so DFS continues.

From `"e"`, the trie needs `"a"`.

A legal neighbor `(1,2)` contains `"a"`:

```text
board path: e -> a
trie prefix: ea
```

From `"ea"`, the trie needs `"t"`.

A legal neighbor `(1,1)` contains `"t"`:

```text
board path: e -> a -> t
trie prefix: eat
```

The trie node for `"eat"` stores the word `"eat"`, so append it.

#### Why `"pea"` Fails Quickly

The root has a child `"p"`, but there is no `"p"` cell on the board.

No DFS path can even consume the first character, so `"pea"` is never found.

#### Why `"rain"` Fails With Prefix Pruning

There is an `"r"` at `(2,3)`, so DFS can start the prefix `"r"`.

The next character must be `"a"`.

If no legal neighboring `"a"` from that `"r"` leads to the later `"i"` and `"n"`, all branches eventually stop when the next board character is missing from the current trie node.

The search does not enumerate arbitrary strings. It only follows strings that are prefixes in the trie.

---

### 9. Correctness Argument

We prove that the algorithm returns exactly the words from `words` that exist on the board.

#### Lemma 1: Every reported word exists on the board.

A word is reported only when DFS reaches a trie node whose `word` field stores that word.

By the DFS invariant, the currently visited board cells spell exactly the trie path from the root to that node.

DFS moves only between adjacent cells and marks cells as visited before exploring deeper, so the path uses legal adjacent moves and does not reuse a cell.

Therefore every reported word is spelled by a valid board path.

#### Lemma 2: Every board-valid target word is reported.

Take any word from `words` that can be spelled by a valid board path:

```text
cell_0, cell_1, ..., cell_k
```

The outer loop starts DFS from every board cell, including `cell_0`.

Because the word was inserted into the trie, each prefix of the word exists as a trie path. Therefore, when DFS follows the valid board path, each next board character is present as a child of the current trie node.

Since the path uses adjacent cells and does not reuse cells, DFS is allowed to make each move in that path.

After consuming the final character, DFS reaches the terminal trie node storing the word, so the algorithm reports it unless it was already reported earlier. In either case, the word appears in the final answer.

#### Lemma 3: No word is reported more than once.

After a word is appended, the algorithm clears that terminal node's `word` field.

Future DFS paths may reach the same trie node, but the word field is now empty, so the word is not appended again.

#### Theorem

By Lemma 1, every returned word is valid. By Lemma 2, every valid target word is returned. By Lemma 3, each returned word appears at most once.

Therefore the algorithm returns exactly the set of target words that exist on the board.

---

### 10. Complexity

Let:

```text
m = number of rows
n = number of columns
W = number of words
S = total number of characters across all words
L = maximum word length
```

#### Building the Trie

Each character of each word is inserted once.

Time:

```text
O(S)
```

Space:

```text
O(S)
```

The trie can have at most one node per inserted character plus the root.

#### DFS Search

From every cell, DFS explores only paths whose spelled prefix exists in the trie.

A loose worst-case upper bound is still exponential in the maximum word length:

```text
O(m * n * 4 * 3^(L - 1))
```

because a board can contain many repeated letters and the trie can contain many matching prefixes.

But compared with checking every word independently, trie pruning removes branches as soon as the current prefix is not present in any candidate word. It also shares work among words with common prefixes.

The recursion depth is at most:

```text
O(L)
```

or at most `O(m * n)` if words can be that long, because a path cannot reuse cells.

---

### 11. Common Pitfalls

#### Confusing Prefixes With Complete Words

A trie node existing means the current string is a prefix of some word.

It does not necessarily mean the current string is itself an answer.

For example, if `"oath"` is in the trie, then `"oa"` is a trie prefix, but `"oa"` should not be reported unless `"oa"` is also in `words`.

That is why terminal markers are separate from child pointers.

#### Forgetting to Restore the Board Cell

If you mark a cell as visited with `"#"`, you must restore the original character before returning.

Otherwise, one DFS path permanently damages the board for later paths.

#### Using a Global Visited Set Without Backtracking

A cell is forbidden only within the current path.

It can be used again by a different word or by a different starting cell.

So if you use a `visited` set, remove the cell when backtracking.

#### Returning Immediately After Finding One Word

Unlike Word Search I, this problem asks for all matching words.

Finding one word does not mean the search from that path is finished.

For example, if both `"oat"` and `"oath"` are in `words`, reaching `"oat"` should report `"oat"` and then continue exploring for `"oath"`.

#### Reporting Duplicates

The same word may be formable through multiple board paths.

Use a result set or clear the terminal `word` field after appending.

Clearing the terminal field is usually efficient and keeps output unique.

#### Mutating the Trie Too Aggressively

Optional pruning can delete a child node after all words under it have been found.

But only delete a child when:

```text
child has no children
and child.word is empty
```

Deleting earlier can remove prefixes still needed by other words.

#### Ignoring Empty Inputs

Production-quality code should handle cases such as:

```text
board = []
words = []
```

LeetCode constraints may rule out some empty cases, but defensive handling is simple.

---

### 12. First-Principles Summary

A board path spells a string one character at a time.

A word list defines which prefixes are worth pursuing.

The brute force approach checks each word separately, but that repeats the same prefix searches many times.

The trie compresses all candidate words into a prefix tree.

DFS walks the board, and the trie walks the word-prefix space at the same time.

The invariant is:

```text
visited board path == trie path to the current node
```

Whenever the next board character is not a trie child, the entire branch is impossible and can be cut off immediately.

Whenever the trie node stores a full word, the current board path is a valid answer.

So the algorithm is not "use a trie because this is a trie problem." It follows from the structure of the search:

```text
many target words
+ shared prefixes
+ board DFS paths that grow by one character
= trie-guided DFS
```

## Implementation

See `solutions/trie/p212_word_search_ii.py`.

## Tests

See `tests/trie/test_p212_word_search_ii.py`.

## Examples

### Example 1
- Input: `{'board': [['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], 'words': ['oath', 'pea', 'eat', 'rain']}`
- Output: `['eat', 'oath']`

### Example 2
- Input: `{'board': [['a', 'b'], ['c', 'd']], 'words': ['abcb']}`
- Output: `[]`

## Follow-up Practice

- Draw the trie for words with shared prefixes such as `"oat"`, `"oath"`, and `"oats"`.
- Trace how the DFS state and trie node move together on the same board path.
- Implement duplicate prevention once with a result set and once by clearing `node.word`.
- Add optional trie-branch pruning after found words and verify it does not remove shared prefixes too early.
