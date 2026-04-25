# 133. Clone Graph

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/clone-graph/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a reference to one node in an undirected connected graph.

Each graph node has two pieces of data:

```text
val       = the node's integer label
neighbors = a list of references to adjacent nodes
```

The task is to return a **deep copy** of the graph.

A deep copy means:

1. Every original node must have a newly allocated clone node.
2. Every original edge must appear between the corresponding clone nodes.
3. No node in the returned graph may be one of the original objects.
4. The shape of the returned graph must be identical to the reachable original graph.

So if the original graph contains an edge:

```text
original A -- original B
```

then the cloned graph must contain:

```text
clone of A -- clone of B
```

The important detail is that we are **not** returning a list of values. We are returning a reference to the cloned version of the input node. From that returned clone, the entire cloned graph must be reachable through `neighbors` links.

If the input node is `None`, there is no graph to copy, so the correct answer is `None`.

---

### 2. Why This Is Trickier Than Copying an Array

For an array, copying is straightforward because elements appear in a simple line:

```text
index 0 -> index 1 -> index 2 -> ...
```

A graph is different because nodes can point back to earlier nodes, sideways to already-seen nodes, or around cycles.

Example:

```text
1 -- 2
|    |
4 -- 3
```

Node `1` has neighbors `2` and `4`.
Node `2` has neighbors `1` and `3`.
Node `3` has neighbors `2` and `4`.
Node `4` has neighbors `1` and `3`.

If we recursively copy neighbors without remembering what we already copied, then copying `1` tries to copy `2`, copying `2` tries to copy `1`, copying `1` tries to copy `2` again, and so on forever.

The graph's cycles are not a special edge case; they are the central reason the problem needs a map.

---

### 3. Start From the Brute Force Baseline

A first idea might be:

```text
To clone a node:
1. Create a new node with the same value.
2. Recursively clone each neighbor.
3. Put those cloned neighbors into the new node's neighbor list.
```

Pseudocode:

```python
def clone(node):
    if node is None:
        return None

    copied = Node(node.val)

    for neighbor in node.neighbors:
        copied.neighbors.append(clone(neighbor))

    return copied
```

This looks natural, but it is wrong for graphs with cycles.

Consider two connected nodes:

```text
1 -- 2
```

Because the graph is undirected:

```text
1.neighbors = [2]
2.neighbors = [1]
```

Calling `clone(1)` calls `clone(2)`, which calls `clone(1)`, which calls `clone(2)`, forever.

Even in an acyclic-looking traversal, this baseline can create multiple different clones for the same original node. That also breaks the graph structure, because all references to the same original node must point to the same clone node.

So the brute-force idea teaches us the key missing ingredient:

> Once an original node has a clone, every future encounter with that original node must reuse the same clone.

---

### 4. The Key Observation: Cloning Is a One-to-One Mapping

A correct clone graph is determined by a mapping:

```text
original node object -> cloned node object
```

For every original node `u`, there must be exactly one clone `copy[u]`.

This mapping solves both hard parts:

1. **Cycles**: if we reach `u` again, we do not recursively clone it again; we return `copy[u]`.
2. **Shared references**: if multiple nodes point to `u`, their cloned neighbors all point to the same `copy[u]`.

The graph clone invariant is:

```text
For every original node already discovered, clone_map[original]
exists and is the unique clone object for that original node.
```

As we traverse edges, we extend this invariant.

If we see a neighbor for the first time, we create its clone and record it. If we see a neighbor that is already in the map, we reuse the existing clone.

That is the whole problem in first-principles form:

```text
Traverse the reachable original graph while maintaining a one-to-one original-to-clone map.
For each original edge u -> v, append clone_map[v] to clone_map[u].neighbors.
```

---

### 5. What Counts as the Same Node?

The map key should be the original node object, not just `node.val`.

LeetCode's version states that node values are unique from `1` to `n`, so using values may appear to work there. But the conceptually correct identity is the node object itself:

```python
clone_map[original_node] = cloned_node
```

Why?

Because we are copying references. The thing that must be preserved is object identity:

```text
If two original neighbor references point to the same original object,
then the two cloned neighbor references must point to the same cloned object.
```

Values help describe the graph, but object identity defines what must be cloned.

---

### 6. DFS Algorithm

Depth-first search is a direct way to express the mapping idea.

Define a helper:

```text
clone_node(original)
```

The helper returns the clone corresponding to `original`.

There are three cases.

#### Case 1: Empty Input

If `original` is `None`, return `None`.

There is no node to clone and no graph to traverse.

#### Case 2: Already Cloned

If `original` is already in `clone_map`, return `clone_map[original]`.

This handles cycles and repeated references.

#### Case 3: First Time Seeing This Node

Create a clone with the same value:

```python
copy = Node(original.val)
```

Immediately store it in the map:

```python
clone_map[original] = copy
```

The word "immediately" matters.

We must record the clone before recursively cloning neighbors. Otherwise, if a neighbor points back to `original`, the recursive call will not know that `original` is already in progress.

Then clone every neighbor:

```python
for neighbor in original.neighbors:
    copy.neighbors.append(clone_node(neighbor))
```

Finally, return `copy`.

---

### 7. Why Recording Before Recursing Is Necessary

Suppose the graph is:

```text
1 -- 2
```

Call `clone_node(1)`.

If we create clone `1'` but do **not** store it before processing neighbors, then:

```text
clone_node(1)
  clone_node(2)
    clone_node(1)
      clone_node(2)
        ...
```

The algorithm still loops forever.

The correct order is:

```text
clone_node(1)
  create 1'
  store 1 -> 1'
  clone_node(2)
    create 2'
    store 2 -> 2'
    clone_node(1)
      1 is already stored, return 1'
```

Now the back edge is handled correctly.

The map is not merely a final result container. It is also the recursion guard.

---

### 8. Detailed Walkthrough of Example 1

Input adjacency list:

```text
[[2, 4], [1, 3], [2, 4], [1, 3]]
```

This represents:

```text
1 -- 2
|    |
4 -- 3
```

More explicitly:

```text
1.neighbors = [2, 4]
2.neighbors = [1, 3]
3.neighbors = [2, 4]
4.neighbors = [1, 3]
```

Start with node `1`.

#### Step 1: Clone Node 1

Create clone `1'` and record:

```text
clone_map = {
  1 -> 1'
}
```

Now process neighbors of `1`: nodes `2` and `4`.

#### Step 2: Clone Neighbor 2

Node `2` is not in the map, so create `2'`:

```text
clone_map = {
  1 -> 1',
  2 -> 2'
}
```

Now process neighbors of `2`: nodes `1` and `3`.

Neighbor `1` is already in the map, so append `1'` to `2'.neighbors`.

Now clone neighbor `3`.

#### Step 3: Clone Node 3

Node `3` is new, so create `3'`:

```text
clone_map = {
  1 -> 1',
  2 -> 2',
  3 -> 3'
}
```

Process neighbors of `3`: nodes `2` and `4`.

Neighbor `2` is already cloned, so append `2'` to `3'.neighbors`.

Now clone neighbor `4`.

#### Step 4: Clone Node 4

Node `4` is new, so create `4'`:

```text
clone_map = {
  1 -> 1',
  2 -> 2',
  3 -> 3',
  4 -> 4'
}
```

Process neighbors of `4`: nodes `1` and `3`.

Both are already cloned:

```text
4'.neighbors = [1', 3']
```

Return `4'` to the call cloning `3`.

Now `3'.neighbors` becomes:

```text
3'.neighbors = [2', 4']
```

Return `3'` to the call cloning `2`.

Now `2'.neighbors` becomes:

```text
2'.neighbors = [1', 3']
```

Return `2'` to the call cloning `1`.

Now `1'.neighbors` starts as:

```text
1'.neighbors = [2']
```

#### Step 5: Finish Node 1's Other Neighbor

The second neighbor of original `1` is original `4`.

Node `4` is already in the map, so return `4'` and append it:

```text
1'.neighbors = [2', 4']
```

The returned graph starts at `1'`, and its adjacency structure is:

```text
1'.neighbors = [2', 4']
2'.neighbors = [1', 3']
3'.neighbors = [2', 4']
4'.neighbors = [1', 3']
```

That matches the original graph's shape, but every node object is newly allocated.

---

### 9. BFS Version of the Same Idea

DFS is not required. Breadth-first search works with the same invariant.

The BFS version is:

1. Create the clone of the starting node.
2. Put the starting original node in a queue.
3. While the queue is not empty:
   - Pop an original node `current`.
   - For each original neighbor:
     - If the neighbor has not been cloned, create its clone and enqueue the original neighbor.
     - Append the neighbor's clone to the current clone's neighbor list.
4. Return the clone of the starting node.

The invariant is identical:

```text
clone_map[x] is the unique clone of original node x.
```

Only the traversal order changes.

DFS often produces the shortest code for this problem, while BFS avoids recursion-depth concerns on very large graphs.

---

### 10. Pseudocode

DFS pseudocode:

```python
def cloneGraph(node):
    if node is None:
        return None

    clone_map = {}

    def clone_node(original):
        if original in clone_map:
            return clone_map[original]

        copied = Node(original.val)
        clone_map[original] = copied

        for neighbor in original.neighbors:
            copied.neighbors.append(clone_node(neighbor))

        return copied

    return clone_node(node)
```

BFS pseudocode:

```python
from collections import deque


def cloneGraph(node):
    if node is None:
        return None

    clone_map = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        current = queue.popleft()

        for neighbor in current.neighbors:
            if neighbor not in clone_map:
                clone_map[neighbor] = Node(neighbor.val)
                queue.append(neighbor)

            clone_map[current].neighbors.append(clone_map[neighbor])

    return clone_map[node]
```

Both versions are correct because both create exactly one clone per reachable original node and copy every neighbor reference into the corresponding clone list.

---

### 11. Correctness

We prove the DFS algorithm returns a deep copy of the graph reachable from the input node.

#### Lemma 1: Every reachable original node gets a clone.

The algorithm starts from the input node and recursively visits each neighbor of every visited node.

A node is reachable exactly when there is some path from the input node to it. By following all neighbor edges recursively, the algorithm eventually encounters every node on every such path.

When a reachable node is encountered for the first time, the algorithm creates a clone and stores it in `clone_map`.

Therefore every reachable original node gets a clone.

#### Lemma 2: Each reachable original node gets exactly one clone.

A clone is created only when the original node is not already in `clone_map`.

Immediately after creating that clone, the algorithm stores:

```text
clone_map[original] = clone
```

Every later encounter with the same original node returns the stored clone instead of creating another one.

Therefore each reachable original node gets exactly one clone.

#### Lemma 3: Every original edge is copied to the cloned graph.

Consider any original edge from `u` to `v`, represented by `v` appearing in `u.neighbors`.

When the algorithm processes `u`, it iterates through every neighbor in `u.neighbors`, including `v`.

It calls `clone_node(v)`, which returns the unique clone of `v`. Then it appends that clone to the neighbor list of the unique clone of `u`.

So the cloned graph contains the edge:

```text
clone(u) -> clone(v)
```

Therefore every original neighbor relationship is copied.

#### Lemma 4: The cloned graph contains no original nodes.

The only objects appended to cloned neighbor lists are return values from `clone_node`.

`clone_node` returns objects that were created by `Node(original.val)` and stored in `clone_map`, except for the empty input case.

It never returns an original node object.

Therefore the returned graph contains only newly allocated clone nodes.

#### Theorem: The algorithm returns a correct deep copy.

By Lemma 1, every reachable original node has a clone.
By Lemma 2, each original node has exactly one clone.
By Lemma 3, every original edge is represented between the corresponding clones.
By Lemma 4, no original node appears in the returned graph.

Therefore the returned node is the root of a graph with the same structure and values as the original reachable graph, but with entirely new node objects. That is exactly a deep copy.

---

### 12. Complexity

Let:

```text
V = number of reachable nodes
E = number of reachable edges
```

For an adjacency-list graph, the algorithm processes each node once and scans each node's neighbor list once.

For an undirected graph, each undirected edge appears twice in neighbor lists, once from each endpoint. This still gives linear total neighbor-list work.

Time complexity:

```text
O(V + E)
```

Space complexity:

```text
O(V)
```

The map stores one clone per node. DFS recursion can also use up to `O(V)` call stack space in the worst case. BFS uses a queue that can hold up to `O(V)` nodes.

---

### 13. Common Pitfalls

#### Pitfall 1: Cloning by Value Instead of Object Identity

Do not think of the map as merely:

```python
value_to_clone[node.val]
```

The conceptual mapping is:

```python
original_object_to_clone[original_node]
```

The graph is made of object references, so object identity is the safest key.

#### Pitfall 2: Adding to the Map Too Late

This is wrong:

```python
copy = Node(node.val)
for neighbor in node.neighbors:
    copy.neighbors.append(clone_node(neighbor))
clone_map[node] = copy
```

It fails on cycles because a neighbor may point back to `node` before `node` has been recorded.

The clone must be stored before processing neighbors.

#### Pitfall 3: Returning the Original Node

A shallow copy mistake is to reuse original neighbor objects:

```python
copy.neighbors = node.neighbors
```

That does not clone the graph. It makes the new node point back into the old graph.

Every neighbor in the clone graph must be a cloned neighbor.

#### Pitfall 4: Forgetting the Empty Graph

If the input is `None`, return `None`.

Trying to read `node.val` or `node.neighbors` before this check will crash.

#### Pitfall 5: Creating Duplicate Clones for Shared Nodes

If node `A` and node `B` both point to node `C`, then the clones of `A` and `B` must both point to the same clone of `C`.

Without a map, it is easy to accidentally create two separate clones of `C`, which changes the graph structure.

#### Pitfall 6: Confusing the Adjacency-List Examples With the Function Input

The examples are shown as adjacency lists because that is easy to print:

```text
[[2, 4], [1, 3], [2, 4], [1, 3]]
```

But the actual function receives a node reference, not the adjacency-list array. The returned value is also a node reference. The judge converts between node references and adjacency lists for testing.

---

### 14. First-Principles Summary

A graph clone is not built by copying values alone. It is built by preserving the reference structure.

The essential invariant is:

```text
Each original node discovered so far has exactly one clone,
and clone_map tells us where that clone is.
```

Once that invariant is maintained, every edge copy becomes local:

```text
for original edge u -> v:
    append clone_map[v] to clone_map[u].neighbors
```

DFS and BFS are only traversal choices. The real idea is the original-to-clone map.

When you see this problem from first principles, it becomes a controlled construction task:

1. Discover nodes reachable from the start.
2. Create one clone per discovered original node.
3. Reuse clones whenever a node is encountered again.
4. Rebuild each neighbor list using cloned neighbors only.
5. Return the clone of the starting node.

That is why the solution is linear in the size of the graph and why it handles cycles naturally.

## Implementation
See `solutions/graph_general/p133_clone_graph.py`.

## Tests
See `tests/graph_general/test_p133_clone_graph.py`.

## Examples

### Example 1
- Input: `{'adjList': [[2, 4], [1, 3], [2, 4], [1, 3]]}`
- Output: `[[2, 4], [1, 3], [2, 4], [1, 3]]`

### Example 2
- Input: `{'adjList': [[]]}`
- Output: `[[]]`

### Example 3
- Input: `{'adjList': []}`
- Output: `[]`

## Follow-up Practice
- Trace the DFS algorithm on a two-node cycle and identify exactly when the map prevents infinite recursion.
- Rewrite the solution with BFS and compare which data structure replaces the recursion stack.
- Explain why assigning `copy.neighbors = node.neighbors` is a shallow copy, not a graph clone.
- Prove that every original edge is copied exactly once per adjacency-list occurrence.
