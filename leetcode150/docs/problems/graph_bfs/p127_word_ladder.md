# 127. Word Ladder

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/word-ladder/
- Official Group: Graph BFS
- Pattern Group: Graph BFS
- Patterns: graph-bfs

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
beginWord
endWord
wordList
```

You may transform one word into another by changing exactly one character at a time.

Every intermediate word must appear in `wordList`.

The goal is to return the length of the shortest valid transformation sequence from `beginWord` to `endWord`.

For example:

```text
beginWord = "hit"
endWord   = "cog"
wordList  = ["hot", "dot", "dog", "lot", "log", "cog"]
```

One valid sequence is:

```text
hit -> hot -> dot -> dog -> cog
```

Its length is `5`, because the sequence contains five words.

The problem does not ask for the sequence itself. It asks only for the number of words in the shortest sequence.

Two details are easy to miss:

```text
Only one character may change per step.
Every word after beginWord must be in wordList.
```

So this is really a shortest-path problem, but the graph is hidden.

Each word is a node.

There is an edge between two words if they differ in exactly one character.

The question becomes:

> In this implicit unweighted graph of words, what is the shortest distance from `beginWord` to `endWord`, counted as number of words in the path?

### 2. Start From the Brute Force Idea

The most direct way to think about the problem is:

1. Start with `beginWord`.
2. Try every possible word in `wordList` that differs by one character.
3. From each of those words, try every next one-character transformation.
4. Continue until `endWord` is found.

If implemented naively, neighbor generation might look like this:

```python
for current_word in frontier:
    for candidate in wordList:
        if candidate differs from current_word by exactly one character:
            try candidate next
```

This is logically correct.

But it repeats a lot of work.

If there are `N` words and each word has length `L`, checking whether two words differ by one character costs `O(L)`. Scanning all words for each visited word therefore costs roughly:

```text
O(N^2 * L)
```

That can be too slow.

The brute-force idea teaches us the right model, though:

```text
current word -> all valid one-letter neighbors
```

The real task is to generate those neighbors efficiently while preserving shortest-path order.

### 3. The Key Observation: Words Are Connected by Patterns

Two words are one transformation apart if they are identical except for one position.

For example:

```text
hot
 dot
```

These differ at the first character only, so they are adjacent.

Instead of comparing every pair of words, replace one position with a wildcard:

```text
hot -> *ot
hot -> h*t
hot -> ho*
```

Any words sharing the same wildcard pattern differ only at that wildcard position.

For example:

```text
hot -> *ot
dot -> *ot
lot -> *ot
```

So from `hot`, all words under pattern `*ot` are valid one-step neighbors.

This gives the central preprocessing idea:

```text
pattern -> list of words that match that pattern
```

For every word of length `L`, generate `L` wildcard patterns.

Then BFS can find neighbors by generating the current word's `L` patterns and looking up the words stored under each pattern.

### 4. Why BFS Is the Right Search

Each transformation changes one character.

Every transformation has the same cost:

```text
one step
```

When all edges have equal cost, BFS is the shortest-path algorithm.

BFS explores in layers:

```text
layer 1: beginWord
layer 2: words reachable in 1 transformation
layer 3: words reachable in 2 transformations
layer 4: words reachable in 3 transformations
...
```

Because BFS finishes all shorter paths before trying longer paths, the first time it reaches `endWord`, that path must be shortest.

This is why DFS is not appropriate here. DFS might find a valid transformation sequence, but it can easily find a long one before a short one.

### 5. BFS State and Invariant

A BFS queue item should store:

```text
(word, length)
```

where:

```text
word   = current word at the end of the transformation sequence
length = number of words in the sequence from beginWord to word
```

At the beginning:

```text
(beginWord, 1)
```

because the sequence already contains `beginWord`.

The visited set stores words that have already been enqueued.

The main invariant is:

```text
When a word is first enqueued with length k,
there exists a valid transformation sequence of length k from beginWord to that word,
and no shorter sequence to that word will ever be needed.
```

This invariant is what makes BFS safe.

Marking a word visited when it is enqueued, not when it is later dequeued, prevents the same word from being placed into the queue many times at the same or greater depth.

### 6. Detailed Algorithm

First, handle the impossible case:

```text
If endWord is not in wordList, return 0.
```

This is required because every transformed word after `beginWord` must come from `wordList`. If `endWord` is absent, no valid sequence can end there.

Then build the wildcard pattern map.

For each word in `wordList`:

```text
For each index i:
    pattern = word with character i replaced by '*'
    add word to pattern_map[pattern]
```

For `"hot"`, this produces:

```text
*ot -> hot
h*t -> hot
ho* -> hot
```

Then run BFS:

1. Put `(beginWord, 1)` into the queue.
2. Mark `beginWord` visited.
3. While the queue is not empty:
   - Pop the next `(word, length)`.
   - If `word == endWord`, return `length`.
   - Generate every wildcard pattern for `word`.
   - For each word stored under those patterns:
     - If it has not been visited, mark it visited and enqueue it with `length + 1`.
4. If BFS ends without reaching `endWord`, return `0`.

A common optimization is to clear a pattern's list after processing it:

```python
pattern_map[pattern] = []
```

This avoids re-scanning the same group many times. The correctness does not depend on this optimization, but it can reduce repeated work.

### 7. Pseudocode

```python
from collections import defaultdict, deque


def ladderLength(beginWord, endWord, wordList):
    word_set = set(wordList)

    if endWord not in word_set:
        return 0

    word_length = len(beginWord)
    pattern_to_words = defaultdict(list)

    for word in word_set:
        for i in range(word_length):
            pattern = word[:i] + "*" + word[i + 1:]
            pattern_to_words[pattern].append(word)

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, length = queue.popleft()

        if word == endWord:
            return length

        for i in range(word_length):
            pattern = word[:i] + "*" + word[i + 1:]

            for neighbor in pattern_to_words[pattern]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))

            pattern_to_words[pattern] = []

    return 0
```

This pseudocode uses the standard wildcard-pattern BFS approach.

There is also a more direct neighbor-generation approach: for each position, try replacing it with every letter `a` through `z`, and check whether the resulting word is in a set. That also works well and has time roughly `O(N * L * 26)` over reached words. The pattern map version makes the hidden graph structure especially explicit.

### 8. Detailed Example Walkthrough

Use the first example:

```text
beginWord = "hit"
endWord   = "cog"
wordList  = ["hot", "dot", "dog", "lot", "log", "cog"]
```

Build pattern groups:

```text
hot -> *ot, h*t, ho*
dot -> *ot, d*t, do*
dog -> *og, d*g, do*
lot -> *ot, l*t, lo*
log -> *og, l*g, lo*
cog -> *og, c*g, co*
```

Important groups include:

```text
*ot: hot, dot, lot
do*: dot, dog
*og: dog, log, cog
lo*: lot, log
```

Now BFS begins.

#### Layer 1

Queue:

```text
(hit, 1)
```

Current word:

```text
hit
```

Patterns:

```text
*it
h*t
hi*
```

Only `h*t` connects to a word in the list:

```text
h*t -> hot
```

Enqueue:

```text
(hot, 2)
```

Visited:

```text
hit, hot
```

#### Layer 2

Current word:

```text
hot
```

Patterns:

```text
*ot
h*t
ho*
```

The pattern `*ot` gives:

```text
hot, dot, lot
```

`hot` is already visited.

New words:

```text
dot, lot
```

Enqueue:

```text
(dot, 3)
(lot, 3)
```

Visited:

```text
hit, hot, dot, lot
```

#### Layer 3

BFS now processes all words at length `3` before any word at length `4`.

First `dot`:

```text
dot -> patterns *ot, d*t, do*
```

The useful pattern is:

```text
do* -> dot, dog
```

`dog` is new, so enqueue:

```text
(dog, 4)
```

Then `lot`:

```text
lot -> patterns *ot, l*t, lo*
```

The useful pattern is:

```text
lo* -> lot, log
```

`log` is new, so enqueue:

```text
(log, 4)
```

Visited now includes:

```text
hit, hot, dot, lot, dog, log
```

#### Layer 4

Process `dog`:

```text
dog -> patterns *og, d*g, do*
```

The pattern `*og` gives:

```text
dog, log, cog
```

`cog` is new.

Enqueue:

```text
(cog, 5)
```

At this point, BFS has found `endWord` at length `5`. Depending on implementation, it may return immediately when generating `cog`, or it may return when `(cog, 5)` is popped from the queue. Both are correct if the BFS level accounting is consistent.

The shortest sequence length is:

```text
5
```

One corresponding sequence is:

```text
hit -> hot -> dot -> dog -> cog
```

### 9. Correctness

We prove that the algorithm returns the length of the shortest valid transformation sequence, or `0` if none exists.

#### Lemma 1: The pattern map generates exactly valid one-letter neighbors.

For a word of length `L`, replacing index `i` with `*` records all words that are equal at every position except possibly `i`.

If two distinct words share such a pattern, they differ only at that one wildcard position, so they are connected by one valid transformation.

Conversely, if two words differ by exactly one character at index `i`, then replacing index `i` with `*` gives the same pattern for both words, so the map will place them in the same group.

Therefore, looking up all wildcard patterns of a word finds exactly the words reachable by one valid transformation, aside from the word itself, which is harmless because it is already visited.

#### Lemma 2: Every enqueued state represents a valid transformation sequence of its stored length.

The initial state `(beginWord, 1)` represents the sequence containing only `beginWord`.

Whenever the algorithm enqueues `(neighbor, length + 1)` from `(word, length)`, the pattern map guarantees that `neighbor` differs from `word` by one character and belongs to `wordList`.

Appending `neighbor` to the valid sequence ending at `word` creates a valid sequence of length `length + 1`.

So every enqueued state is valid.

#### Lemma 3: The first time a word is enqueued, it has the shortest possible length from `beginWord`.

BFS processes states in nondecreasing `length` order because every queue expansion adds exactly `1` to the length and the queue is first-in, first-out.

Suppose a word were first enqueued with length `k`, but there existed a shorter valid sequence of length `< k`.

Then the predecessor of that word on the shorter sequence would have been processed at an earlier BFS layer, and the word would have been discovered earlier.

That contradicts the assumption that its first enqueue length was `k`.

Therefore, first discovery gives the shortest length.

#### Theorem: The algorithm returns the correct answer.

If the algorithm returns a positive length, it returns when `endWord` is reached by BFS. By Lemma 2, that length corresponds to a valid transformation sequence. By Lemma 3, it is the shortest possible such length.

If the algorithm returns `0`, BFS has exhausted every reachable word without finding `endWord`. Since Lemma 1 guarantees that BFS considered all valid one-letter transformations, no valid transformation sequence exists.

Therefore, the algorithm is correct.

### 10. Complexity

Let:

```text
N = number of words in wordList
L = length of each word
```

Building the pattern map creates `L` patterns per word.

Each pattern construction costs up to `O(L)` in Python because slicing creates new strings.

So preprocessing costs:

```text
O(N * L^2)
```

BFS generates `L` patterns for each visited word, also with slicing cost `O(L)` per pattern.

The total size of all pattern groups is `O(N * L)`, and with the clearing optimization each group list is scanned at most once.

So BFS is commonly described as:

```text
O(N * L^2)
```

with Python slicing included.

Space usage is:

```text
O(N * L)
```

for the pattern map, plus:

```text
O(N)
```

for the queue and visited set.

If we ignore the cost of string slicing and treat pattern creation as `O(L)`, the same high-level result is still dominated by storing and processing `L` patterns for each word.

### 11. Common Pitfalls

#### Forgetting the `endWord not in wordList` case

If `endWord` is absent from `wordList`, the answer must be `0`.

The transformation sequence cannot end at a word that is not allowed.

#### Counting edges instead of words

The output is the number of words in the sequence, not the number of transformations.

For:

```text
hit -> hot -> dot -> dog -> cog
```

there are `4` transformations but `5` words.

So BFS should start with length `1`, not `0`, unless the implementation carefully adds one at the end.

#### Marking visited too late

Mark a neighbor visited when it is enqueued.

If you wait until it is dequeued, multiple parents in the same BFS layer can enqueue the same word repeatedly.

That does not usually change correctness, but it can waste substantial time and memory.

#### Comparing every word pair repeatedly

A naive BFS that scans the whole word list for every current word can become `O(N^2 * L)`.

The wildcard pattern map avoids this by turning neighbor lookup into pattern lookup.

#### Treating the graph as directed

Transformations are naturally undirected.

If `hot` can become `dot`, then `dot` can become `hot`.

Visited handling prevents cycling back and forth.

#### Returning when a word is seen in a pattern without checking visited or level

It is safe to return immediately when `neighbor == endWord` if you return `length + 1` from the current BFS state.

It is also safe to return when `endWord` is popped from the queue with its stored length.

Mixing these two styles can create off-by-one errors.

### 12. First-Principles Summary

Word Ladder is not fundamentally about strings.

It is about shortest paths in an implicit unweighted graph.

The first-principles model is:

```text
word = graph node
one-character transformation = graph edge
shortest transformation sequence = BFS shortest path
```

The important implementation idea is to avoid building the full graph by pairwise comparison.

Wildcard patterns reveal adjacency directly:

```text
hot, dot, lot share *ot
```

BFS then maintains a simple invariant:

```text
The first time a word is discovered, its stored length is the shortest valid sequence length to that word.
```

Once this invariant is clear, the algorithm is a direct translation:

```text
precompute wildcard groups
BFS from beginWord
expand one-letter neighbors
return the first length that reaches endWord
```

## Implementation
See `solutions/graph_bfs/p127_word_ladder.py`.

## Tests
See `tests/graph_bfs/test_p127_word_ladder.py`.

## Examples

### Example 1
- Input: `{'beginWord': 'hit', 'endWord': 'cog', 'wordList': ['hot', 'dot', 'dog', 'lot', 'log', 'cog']}`
- Output: `5`

### Example 2
- Input: `{'beginWord': 'hit', 'endWord': 'cog', 'wordList': ['hot', 'dot', 'dog', 'lot', 'log']}`
- Output: `0`

## Follow-up Practice
- Trace BFS levels for `hit -> cog` and write the queue contents after each level.
- Implement both neighbor-generation strategies: wildcard pattern map and 26-letter mutation.
- Explain why BFS returns the shortest sequence but DFS does not.
- Add a guard for `endWord not in wordList` before doing any preprocessing.
