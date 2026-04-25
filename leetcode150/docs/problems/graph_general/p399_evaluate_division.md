# 399. Evaluate Division

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/evaluate-division/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

We are given equations such as:

```text
a / b = 2.0
b / c = 3.0
```

Then we are asked queries such as:

```text
a / c = ?
b / a = ?
a / e = ?
a / a = ?
```

The variables are strings. They are not numbers by themselves. The only information we know about them comes from the ratios in `equations`.

For the first two equations:

```text
a / b = 2.0
b / c = 3.0
```

we can combine them:

```text
a / c
= (a / b) * (b / c)
= 2.0 * 3.0
= 6.0
```

So the answer to `a / c` is `6.0`.

For `b / a`, we reverse the first equation:

```text
b / a = 1 / (a / b) = 1 / 2.0 = 0.5
```

For `a / e`, there is no known relationship involving `e`, so the answer is `-1.0`.

For `a / a`, the answer is `1.0` only if `a` is a known variable. Any known nonzero quantity divided by itself is `1.0`. But if the query is `x / x` and `x` never appears in any equation, the problem expects `-1.0`, because `x` is not part of the known system at all.

So the real problem is:

> Given some known pairwise ratios, answer whether two variables are connected by a chain of ratios, and if so multiply the ratios along that chain.

This is not primarily an arithmetic problem. It is a connectivity problem with multiplication attached to each connection.

---

### 2. The Brute Force Baseline

A direct way to answer a query `x / y` is to try to derive it from equations by repeatedly substituting known ratios.

For example, if we know:

```text
a / b = 2
b / c = 3
c / d = 4
```

then:

```text
a / d = (a / b) * (b / c) * (c / d)
      = 2 * 3 * 4
      = 24
```

The brute-force thought process is:

1. For each query, start at the numerator variable.
2. Try every equation that contains the current variable.
3. Move to another variable whose ratio is known.
4. Keep multiplying ratios along the way.
5. Stop if the denominator variable is reached.

This is already very close to graph search. The expensive part is how we find the next usable equations.

If we scan the entire `equations` list every time we stand on a variable, a single query can repeatedly inspect many irrelevant equations. With many queries, that becomes wasteful.

Conceptually:

```python
for query in queries:
    search from query[0] to query[1]
    each search step scans every equation to find neighbors
```

If there are `E` equations and many possible search states, this can be much slower than necessary.

The first improvement is to store the equations in the form that the search actually needs: for each variable, list the variables directly reachable from it and the ratio needed to move there.

That gives us a graph.

---

### 3. Turning Equations Into a Graph

Each variable is a graph node.

Each equation creates two directed weighted edges.

If:

```text
a / b = value
```

then we know both:

```text
a -> b has weight value
b -> a has weight 1 / value
```

Why directed edges?

Because `a / b` and `b / a` are different values unless the ratio is `1`.

For:

```text
a / b = 2.0
```

the graph contains:

```text
a --2.0--> b
b --0.5--> a
```

For:

```text
b / c = 3.0
```

the graph also contains:

```text
b --3.0--> c
c --1/3--> b
```

Together:

```text
a --2.0--> b --3.0--> c
c --1/3--> b --0.5--> a
```

Now a query asks for the product of edge weights along a path.

For `a / c`:

```text
a --2.0--> b --3.0--> c
```

The product is:

```text
2.0 * 3.0 = 6.0
```

For `c / a`:

```text
c --1/3--> b --0.5--> a
```

The product is:

```text
(1/3) * 0.5 = 1/6
```

So every query becomes:

> Is there a path from the numerator node to the denominator node? If yes, multiply the edge weights on that path. If no, return `-1.0`.

---

### 4. The Weighted-Graph Invariant

The important invariant is not just that two variables are connected. It is what the path product means.

For every edge:

```text
u --w--> v
```

the weight means:

```text
u / v = w
```

For any path:

```text
x -> n1 -> n2 -> ... -> y
```

the product of edge weights means:

```text
x / n1 * n1 / n2 * n2 / ... * ... / y
```

All middle variables cancel:

```text
x / y
```

That cancellation is the whole reason the graph model works.

For example:

```text
a / b = 2
b / c = 3
c / d = 4
```

Path product:

```text
(a / b) * (b / c) * (c / d)
```

After cancellation:

```text
a / d
```

Numerically:

```text
2 * 3 * 4 = 24
```

This gives the invariant used during DFS or BFS:

> When the search is currently at node `cur` with accumulated value `acc`, `acc` equals `start / cur`.

At the beginning:

```text
cur = start
acc = 1.0
```

because:

```text
start / start = 1.0
```

If we move from `cur` to `next` using an edge weight `w`, then:

```text
cur / next = w
```

and the new accumulated value is:

```text
acc * w
= (start / cur) * (cur / next)
= start / next
```

So the invariant is preserved after every step.

When `cur` becomes the query denominator `target`, the accumulated value is:

```text
start / target
```

which is exactly the query answer.

---

### 5. Why Any Found Path Is Enough

The problem input is designed so that equations are consistent. That means if there are multiple paths between the same two variables, they imply the same ratio.

For example, if both paths exist:

```text
a -> b -> c
a -> d -> c
```

then both path products represent `a / c`, so they should agree.

Because of that, we do not need to find a shortest path, cheapest path, or best path. We only need to find any path from the numerator to the denominator.

This is why ordinary DFS or BFS is enough.

DFS answers the question:

```text
Can I reach target from start, and what product did I accumulate on the way?
```

BFS answers the same question, just in a different traversal order.

Neither traversal is optimizing anything. Both are only searching for a connected route while maintaining the product invariant.

---

### 6. Detailed Algorithm

Build an adjacency map:

```text
graph[variable] = list of (neighbor, ratio)
```

For each equation:

```text
equations[i] = [a, b]
values[i] = value
```

add:

```text
graph[a].append((b, value))
graph[b].append((a, 1 / value))
```

Then answer each query `[start, target]`:

1. If `start` or `target` is missing from the graph, return `-1.0`.
2. If `start == target`, return `1.0` because the variable is known.
3. Run DFS or BFS from `start`.
4. Carry an accumulated product initialized to `1.0`.
5. When moving across edge `(cur, next, weight)`, pass `acc * weight` to `next`.
6. If `target` is reached, return the accumulated product.
7. If the search finishes without reaching `target`, return `-1.0`.

The visited set is necessary because equations can form cycles:

```text
a / b = 2
b / c = 3
c / a = 1/6
```

Without `visited`, DFS or BFS could loop forever by repeatedly walking around the cycle.

---

### 7. Pseudocode

One clean DFS version is:

```python
from collections import defaultdict


def calcEquation(equations, values, queries):
    graph = defaultdict(list)

    for (a, b), value in zip(equations, values):
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))

    def dfs(cur, target, acc, visited):
        if cur == target:
            return acc

        visited.add(cur)

        for neighbor, weight in graph[cur]:
            if neighbor in visited:
                continue

            result = dfs(neighbor, target, acc * weight, visited)
            if result != -1.0:
                return result

        return -1.0

    answers = []

    for start, target in queries:
        if start not in graph or target not in graph:
            answers.append(-1.0)
        else:
            answers.append(dfs(start, target, 1.0, set()))

    return answers
```

The same idea can be written with BFS:

```python
from collections import defaultdict, deque


def calcEquation(equations, values, queries):
    graph = defaultdict(list)

    for (a, b), value in zip(equations, values):
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))

    def bfs(start, target):
        queue = deque([(start, 1.0)])
        visited = {start}

        while queue:
            cur, acc = queue.popleft()

            if cur == target:
                return acc

            for neighbor, weight in graph[cur]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, acc * weight))

        return -1.0

    answers = []

    for start, target in queries:
        if start not in graph or target not in graph:
            answers.append(-1.0)
        else:
            answers.append(bfs(start, target))

    return answers
```

Both versions rely on the same invariant. The choice between them is mostly style for this problem.

---

### 8. Detailed Example Walkthrough

Use Example 1:

```text
equations = [["a", "b"], ["b", "c"]]
values    = [2.0, 3.0]
queries   = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
```

Build the graph.

From:

```text
a / b = 2.0
```

add:

```text
a -> b, weight 2.0
b -> a, weight 0.5
```

From:

```text
b / c = 3.0
```

add:

```text
b -> c, weight 3.0
c -> b, weight 1/3
```

Adjacency list:

```text
a: [(b, 2.0)]
b: [(a, 0.5), (c, 3.0)]
c: [(b, 0.333...)]
```

#### Query `a / c`

Start:

```text
cur = a
acc = 1.0
```

The invariant says:

```text
acc = a / cur
```

At `a`, move to `b` using weight `2.0`:

```text
new_acc = 1.0 * 2.0 = 2.0
```

Now:

```text
cur = b
acc = 2.0
```

This means:

```text
a / b = 2.0
```

At `b`, skip `a` if already visited, then move to `c` using weight `3.0`:

```text
new_acc = 2.0 * 3.0 = 6.0
```

Now:

```text
cur = c
acc = 6.0
```

The target is reached, so:

```text
a / c = 6.0
```

#### Query `b / a`

Start:

```text
cur = b
acc = 1.0
```

At `b`, there is a direct edge to `a` with weight `0.5`:

```text
new_acc = 1.0 * 0.5 = 0.5
```

The target is reached:

```text
b / a = 0.5
```

#### Query `a / e`

Variable `e` does not appear in the graph.

There is no known value for `e`, so return:

```text
-1.0
```

#### Query `a / a`

Variable `a` appears in the graph.

Because it is known:

```text
a / a = 1.0
```

Return:

```text
1.0
```

#### Query `x / x`

Even though any real nonzero quantity divided by itself is `1`, the variable `x` is unknown to the equation system.

The problem asks us to evaluate using only known equations. Since `x` never appears, return:

```text
-1.0
```

Final answer:

```text
[6.0, 0.5, -1.0, 1.0, -1.0]
```

---

### 9. Correctness

We prove that the algorithm returns the correct value for every query.

#### Lemma 1: Every graph edge represents a valid equation ratio.

For each input equation `a / b = value`, the algorithm adds edge `a -> b` with weight `value`, which directly represents `a / b`. It also adds edge `b -> a` with weight `1 / value`, which represents the reciprocal ratio `b / a`. Therefore every graph edge represents a valid known ratio.

#### Lemma 2: During a search for `start / target`, whenever the algorithm visits a node `cur` with accumulated value `acc`, `acc = start / cur`.

Initially, the search starts at `start` with `acc = 1.0`, and `start / start = 1.0`, so the invariant holds.

Assume the invariant holds at `cur`, so `acc = start / cur`. If the algorithm traverses an edge `cur -> next` with weight `w`, then by Lemma 1, `w = cur / next`. The new accumulated value is:

```text
acc * w = (start / cur) * (cur / next) = start / next
```

So the invariant also holds at `next`. By induction, the invariant holds for every visited node.

#### Lemma 3: If the algorithm reaches `target`, the returned value equals `start / target`.

By Lemma 2, when `cur == target`, the accumulated value satisfies:

```text
acc = start / target
```

The algorithm returns exactly this value, so the returned value is correct.

#### Lemma 4: If the algorithm does not reach `target`, no known chain of equations can evaluate `start / target`.

DFS or BFS explores every node reachable from `start` through graph edges, while `visited` only prevents revisiting nodes already explored. If `target` is not reached, then `target` is not in the same connected component as `start`. Therefore there is no path of known ratios from `start` to `target`, so the query cannot be evaluated from the input equations.

#### Theorem: The algorithm returns the correct answer for every query.

If either query variable is missing from the graph, no equation mentions it, so the query is not evaluable and the algorithm correctly returns `-1.0`. Otherwise, if a path exists, Lemma 3 shows that the algorithm returns the correct ratio. If no path exists, Lemma 4 shows that the query cannot be evaluated, so returning `-1.0` is correct. Therefore every query answer is correct.

---

### 10. Complexity

Let:

```text
E = number of equations
V = number of distinct variables
Q = number of queries
```

Building the graph adds two edges for every equation:

```text
Time:  O(E)
Space: O(V + E)
```

For one query, DFS or BFS may visit every variable and every edge in the connected component. In the worst case, that is:

```text
Time per query: O(V + E)
```

For all queries:

```text
Total time: O(E + Q * (V + E))
```

The extra space during one search is the visited set plus recursion stack or queue:

```text
Extra search space: O(V)
```

The stored graph remains:

```text
Graph space: O(V + E)
```

---

### 11. Common Pitfalls

- Forgetting reciprocal edges. If `a / b = 2`, then `b / a = 0.5` is also needed.
- Returning `1.0` for an unknown self-query like `x / x`. The correct result is `-1.0` unless `x` appears in the graph.
- Adding a variable to `visited` too late. In cyclic graphs, this can cause repeated work or infinite recursion.
- Multiplying in the wrong direction. For edge `u -> v`, the weight must mean `u / v`, not `v / u`.
- Treating the task as a shortest-path problem. The edge weights are ratios, not distances; any valid path is enough because the equations are consistent.
- Reusing one `visited` set across different queries. Each query needs a fresh search state.
- Comparing floating-point results too strictly in custom tests. Products like `1 / 3` may produce normal floating-point rounding.

---

### 12. First-Principles Summary

The problem gives relationships between variables, not standalone variable values.

An equation:

```text
a / b = value
```

is exactly a directed weighted relationship:

```text
a -> b = value
b -> a = 1 / value
```

A chain of equations is a path. Multiplying the edge weights along that path cancels the intermediate variables:

```text
(a / b) * (b / c) * (c / d) = a / d
```

Therefore each query is solved by graph search:

1. Build the weighted graph from equations.
2. For each query, check that both variables are known.
3. Search from numerator to denominator.
4. Carry the product of ratios along the path.
5. Return the product if the denominator is reached; otherwise return `-1.0`.

The central invariant is:

```text
accumulated_product = query_start / current_node
```

Once that invariant is clear, the implementation is just DFS or BFS with multiplication.

## Implementation
See `solutions/graph_general/p399_evaluate_division.py`.

## Tests
See `tests/graph_general/test_p399_evaluate_division.py`.

## Examples

### Example 1
- Input: `{'equations': [['a', 'b'], ['b', 'c']], 'values': [2.0, 3.0], 'queries': [['a', 'c'], ['b', 'a'], ['a', 'e'], ['a', 'a'], ['x', 'x']]}`
- Output: `[6.0, 0.5, -1.0, 1.0, -1.0]`

### Example 2
- Input: `{'equations': [['a', 'b'], ['b', 'c'], ['bc', 'cd']], 'values': [1.5, 2.5, 5.0], 'queries': [['a', 'c'], ['c', 'b'], ['bc', 'cd'], ['cd', 'bc']]}`
- Output: `[3.75, 0.4, 5.0, 0.2]`

### Example 3
- Input: `{'equations': [['a', 'b']], 'values': [0.5], 'queries': [['a', 'b'], ['b', 'a'], ['a', 'c'], ['x', 'y']]}`
- Output: `[0.5, 2.0, -1.0, -1.0]`

## Follow-up Practice

- Trace a query where the answer uses more than one edge.
- Explain why multiplying along a path cancels intermediate variables.
- Compare DFS and BFS for one query and confirm they maintain the same invariant.
- Add a disconnected component and verify that cross-component queries return `-1.0`.
