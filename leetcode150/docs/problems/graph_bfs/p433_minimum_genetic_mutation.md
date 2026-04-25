# 433. Minimum Genetic Mutation

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/minimum-genetic-mutation/
- Official Group: Graph BFS
- Pattern Group: Graph BFS
- Patterns: graph-bfs

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
startGene = an 8-character gene string
endGene   = another 8-character gene string
bank      = a list of allowed gene strings
```

Each gene string contains only these four characters:

```text
A, C, G, T
```

One mutation changes exactly one position of the current gene into one of the other three possible letters.

For example:

```text
AACCGGTT -> AACCGGTA
```

is one mutation, because only the last character changed:

```text
T -> A
```

But not every one-character change is allowed. Every intermediate gene after a mutation must appear in `bank`.

The task is:

> Find the minimum number of valid one-character mutations needed to transform `startGene` into `endGene`.

If no sequence of valid mutations can reach `endGene`, return `-1`.

So the problem is not asking whether two strings are similar in general. It is asking for the shortest path from one gene string to another, where each move changes exactly one character and lands on a gene that is explicitly allowed by `bank`.

---

### 2. Start From the Brute Force Idea

A direct way to think about the problem is:

1. Start at `startGene`.
2. Try every possible one-character mutation.
3. Keep only mutations that appear in `bank`.
4. From each valid mutation, try every next mutation.
5. Continue until `endGene` is found.

This brute force search is conceptually correct, but it has two immediate dangers.

First, it can revisit the same gene repeatedly:

```text
AAAAACCC -> AAAAACCA -> AAAAACCC -> AAAAACCA -> ...
```

Without a visited set, the search can cycle forever or do the same work many times.

Second, if we search in an arbitrary depth-first order, we might find a valid path but not the shortest one. For example, suppose there are two possible routes:

```text
start -> x -> end
start -> a -> b -> c -> end
```

A depth-first search might discover the longer route first and return `4`, even though the correct minimum is `2`.

The brute force idea needs one extra principle:

> Because every mutation has the same cost, search by number of mutations, not by arbitrary path order.

That principle leads directly to breadth-first search.

---

### 3. Key Observation: This Is an Unweighted Shortest Path

Imagine every valid gene as a node in a graph.

Two nodes have an edge between them if the two gene strings differ in exactly one position.

For example:

```text
AACCGGTT -- AACCGGTA
```

There is an edge because one mutation transforms one into the other.

Every edge has the same cost:

```text
one mutation
```

Therefore, the problem becomes:

```text
Find the shortest path length from startGene to endGene in an unweighted graph.
```

Breadth-first search is the natural tool for this because BFS explores all states at distance `0`, then all states at distance `1`, then all states at distance `2`, and so on.

So the first time BFS reaches `endGene`, the number of levels already processed is the minimum possible number of mutations.

---

### 4. Why We Do Not Need to Build the Whole Graph First

One possible implementation is to compare every pair of genes in `bank` and connect pairs that differ by one character.

That works, but it is unnecessary.

Each gene has length `8`, and each position can be changed to one of four letters. So from any current gene, we can generate all possible one-step mutations directly:

```text
for each index in the gene:
    replace that character with A, C, G, or T
```

For an 8-character gene, this creates at most:

```text
8 positions * 4 letters = 32 candidates
```

Some candidates are the same as the original gene because the replacement letter may equal the current letter. Those can be ignored.

Some candidates are not in `bank`. Those are invalid and must be ignored.

This means we can treat the graph as implicit:

```text
current gene -> generated valid neighboring genes
```

We only generate neighbors when BFS actually visits a gene.

---

### 5. BFS State and Invariant

The BFS queue stores genes whose outgoing one-mutation neighbors still need to be explored.

Each queued item needs to know:

```text
gene  = current gene string
steps = number of valid mutations used to reach this gene from startGene
```

The main invariant is:

```text
When (gene, steps) is popped from the queue, steps is the minimum number of mutations needed to reach gene.
```

This invariant is true because BFS enqueues states in increasing distance order.

We also maintain a visited set:

```text
visited = genes already enqueued
```

A gene should be marked visited when it is enqueued, not when it is popped. That prevents the same gene from being added to the queue multiple times by different parents at the same BFS level.

The visited invariant is:

```text
Each gene is enqueued at most once, and the first enqueue gives its shortest distance.
```

---

### 6. Important Validity Rule: The Bank Controls Reachability

A mutation is valid only if the resulting gene is in `bank`.

That has an important consequence:

```text
If endGene is not in bank, the answer is usually impossible and should be -1.
```

Why? Because the final mutation must land on `endGene`, and every gene reached after a mutation must be in `bank`.

The only special case is when `startGene == endGene`. In that situation, zero mutations are needed. Depending on implementation style, you can return `0` immediately before checking whether `endGene` is in `bank`.

For the usual LeetCode inputs, `startGene` and `endGene` are distinct, so an early check:

```python
if endGene not in bank_set:
    return -1
```

is safe after handling the zero-mutation case.

---

### 7. Detailed Algorithm

1. If `startGene == endGene`, return `0`.

2. Convert `bank` into a set:

```python
bank_set = set(bank)
```

This lets us test whether a candidate mutation is valid in constant time.

3. If `endGene` is not in `bank_set`, return `-1`.

4. Initialize BFS:

```python
queue = deque([(startGene, 0)])
visited = {startGene}
```

5. While the queue is not empty:

   1. Pop the front state:

   ```python
   gene, steps = queue.popleft()
   ```

   2. If `gene == endGene`, return `steps`.

   3. Generate every one-character mutation of `gene`.

   4. For each candidate mutation:

      - skip it if it is not in `bank_set`
      - skip it if it is already in `visited`
      - otherwise mark it visited and enqueue it with `steps + 1`

6. If BFS finishes without reaching `endGene`, return `-1`.

---

### 8. Pseudocode

```python
from collections import deque


def minMutation(startGene, endGene, bank):
    if startGene == endGene:
        return 0

    bank_set = set(bank)

    if endGene not in bank_set:
        return -1

    letters = "ACGT"
    queue = deque([(startGene, 0)])
    visited = {startGene}

    while queue:
        gene, steps = queue.popleft()

        if gene == endGene:
            return steps

        for index in range(len(gene)):
            for letter in letters:
                if letter == gene[index]:
                    continue

                next_gene = gene[:index] + letter + gene[index + 1:]

                if next_gene not in bank_set:
                    continue

                if next_gene in visited:
                    continue

                visited.add(next_gene)
                queue.append((next_gene, steps + 1))

    return -1
```

The implementation can also return as soon as it generates `endGene`, but checking when a gene is popped keeps the BFS invariant especially clear:

```text
popped step count = shortest distance for that gene
```

---

### 9. Example Walkthrough 1

Input:

```text
startGene = "AACCGGTT"
endGene   = "AACCGGTA"
bank      = ["AACCGGTA"]
```

Start BFS:

```text
queue   = [("AACCGGTT", 0)]
visited = {"AACCGGTT"}
```

Pop:

```text
gene = "AACCGGTT"
steps = 0
```

Generate one-character mutations. The only generated mutation that appears in the bank is:

```text
AACCGGTA
```

Enqueue it:

```text
queue   = [("AACCGGTA", 1)]
visited = {"AACCGGTT", "AACCGGTA"}
```

Pop:

```text
gene = "AACCGGTA"
steps = 1
```

This is `endGene`, so return:

```text
1
```

The answer is `1` because one valid mutation transforms the start gene into the end gene.

---

### 10. Example Walkthrough 2

Input:

```text
startGene = "AACCGGTT"
endGene   = "AAACGGTA"
bank      = ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
```

Initial state:

```text
queue   = [("AACCGGTT", 0)]
visited = {"AACCGGTT"}
```

#### Level 0

Pop:

```text
AACCGGTT, steps = 0
```

Valid one-mutation neighbors in the bank:

```text
AACCGGTA
```

Enqueue:

```text
queue = [("AACCGGTA", 1)]
```

#### Level 1

Pop:

```text
AACCGGTA, steps = 1
```

Valid one-mutation neighbors in the bank include:

```text
AAACGGTA
AACCGCTA
```

Both are one mutation away from `AACCGGTA` and both are in the bank.

Enqueue unvisited neighbors:

```text
queue = [("AAACGGTA", 2), ("AACCGCTA", 2)]
```

#### Level 2

Pop:

```text
AAACGGTA, steps = 2
```

This equals `endGene`, so return:

```text
2
```

The shortest valid sequence is:

```text
AACCGGTT -> AACCGGTA -> AAACGGTA
```

It uses two mutations, so the answer is `2`.

---

### 11. Correctness Argument

We prove that the algorithm returns the minimum number of mutations needed to transform `startGene` into `endGene`, or `-1` if no valid sequence exists.

#### Lemma 1: Every enqueued gene is reachable by a valid mutation sequence.

The BFS starts by enqueueing `startGene` with distance `0`, which is reachable using zero mutations.

Whenever the algorithm enqueues `next_gene` from `gene`, it only does so after generating `next_gene` by changing exactly one character of `gene` and checking that `next_gene` is in `bank_set`.

So `next_gene` is reachable from `gene` by one valid mutation. Since `gene` was already reachable, `next_gene` is reachable by a valid mutation sequence of length `steps + 1`.

Therefore every enqueued gene is reachable by a valid sequence.

#### Lemma 2: When a gene is popped from the queue with value `steps`, `steps` is the minimum number of mutations needed to reach it.

BFS begins with all states at distance `0`.

When a state at distance `d` is processed, all newly enqueued neighbors receive distance `d + 1`.

Because the queue is first-in, first-out, all states at distance `d` are popped before any state at distance `d + 1`.

Also, a gene is marked visited when it is first enqueued, so the first discovered distance is the only distance recorded for that gene.

Thus, when a gene is popped, no shorter path to that gene can still be undiscovered. Its `steps` value is minimal.

#### Lemma 3: If the algorithm returns `steps` for `endGene`, that value is the minimum possible answer.

The algorithm returns only when `endGene` is popped from the BFS queue.

By Lemma 2, the `steps` associated with that popped state is the minimum number of mutations needed to reach `endGene`.

Therefore the returned value is optimal.

#### Lemma 4: If the algorithm returns `-1`, no valid mutation sequence reaches `endGene`.

The algorithm explores every reachable gene that can be obtained through valid one-character mutations landing in `bank_set`.

If the queue becomes empty, every reachable valid gene has been processed and none was `endGene`.

Therefore no valid mutation sequence from `startGene` to `endGene` exists.

#### Conclusion

By Lemmas 1 through 4, the algorithm returns exactly the minimum number of valid mutations when such a sequence exists, and returns `-1` otherwise.

---

### 12. Complexity

Let:

```text
n = number of genes in bank
L = length of each gene
```

For this problem, `L` is fixed at `8`, but it is useful to keep it symbolic.

Each gene is enqueued at most once because of `visited`.

For each popped gene, we generate:

```text
L * 4
```

candidate strings.

Creating a candidate with slicing costs `O(L)` in Python, because it builds a new string.

So the time complexity is:

```text
O(n * L * 4 * L) = O(n * L^2)
```

Since the alphabet size `4` is constant, it is usually written as:

```text
O(n * L^2)
```

Because `L = 8` in this problem, this behaves like linear time in the number of bank genes.

The space complexity is:

```text
O(n)
```

for the bank set, visited set, and BFS queue.

---

### 13. Common Pitfalls

#### Forgetting that `endGene` must be in the bank

If `endGene` is not in `bank`, no final valid mutation can land there. Return `-1` unless `startGene == endGene` is handled as a zero-mutation case.

#### Using DFS and returning the first found path

DFS can find a path, but not necessarily the shortest path. This problem asks for the minimum number of mutations, so BFS is the safer direct fit.

#### Marking visited too late

If a gene is marked visited only after popping, multiple parents can enqueue the same gene. That wastes work and can make reasoning about distances messier.

Mark it visited when enqueueing.

#### Generating mutations that do not change the gene

Replacing a character with itself creates the same gene again. Skip those replacements:

```python
if letter == gene[index]:
    continue
```

#### Comparing only against the current BFS frontier

A candidate is valid because it appears in `bank`, not because it appears near the current queue. The bank is the full dictionary of allowed states.

#### Off-by-one step counting

The start gene has distance `0`, not `1`. A neighbor of the start gene has distance `1`.

A good rule is:

```text
enqueue neighbor with current steps + 1
```

---

### 14. First-Principles Summary

The problem gives strings, but the structure is a graph.

A gene is a node.

A valid one-character mutation is an edge.

The bank defines which nodes are allowed to be visited.

Because every edge costs exactly one mutation, the shortest valid mutation sequence is the shortest path in an unweighted graph.

Breadth-first search is correct because it explores by distance:

```text
0 mutations, then 1 mutation, then 2 mutations, ...
```

The queue stores the current boundary of reachable genes, and the visited set ensures each gene is processed only at its earliest possible distance.

Once `endGene` is reached, BFS has already ruled out every shorter sequence, so the current step count is the answer.

## Implementation
See `solutions/graph_bfs/p433_minimum_genetic_mutation.py`.

## Tests
See `tests/graph_bfs/test_p433_minimum_genetic_mutation.py`.

## Examples

### Example 1
- Input: `{'startGene': 'AACCGGTT', 'endGene': 'AACCGGTA', 'bank': ['AACCGGTA']}`
- Output: `1`

### Example 2
- Input: `{'startGene': 'AACCGGTT', 'endGene': 'AAACGGTA', 'bank': ['AACCGGTA', 'AACCGCTA', 'AAACGGTA']}`
- Output: `2`

## Follow-up Practice
- Trace the BFS queue level by level on both examples.
- Write a helper that generates all valid one-character mutations of a gene.
- Explain why the first time BFS reaches `endGene` must be optimal.
- Compare this problem to Word Ladder: both are shortest paths over strings with one-character transitions.
