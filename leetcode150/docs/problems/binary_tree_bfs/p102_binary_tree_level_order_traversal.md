# 102. Binary Tree Level Order Traversal

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/binary-tree-level-order-traversal/
- Official Group: Binary Tree BFS
- Pattern Group: Binary Tree BFS
- Patterns: binary-tree-bfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, return the node values grouped by depth from top to bottom.

That means the answer is not one flat traversal like:

```text
[3, 9, 20, 15, 7]
```

Instead, values that live on the same level must be placed in the same inner list:

```text
[[3], [9, 20], [15, 7]]
```

For this tree:

```text
        3
       / \
      9   20
         /  \
        15   7
```

The levels are:

```text
level 0: 3
level 1: 9, 20
level 2: 15, 7
```

So the real task is:

> Visit every node exactly once, but separate the visited values by distance from the root.

The phrase **level order** means:

```text
all nodes at depth 0,
then all nodes at depth 1,
then all nodes at depth 2,
...
```

The ordering inside each level is left to right, because for each node we consider its left child before its right child.

---

### 2. Start From a Baseline Idea

A direct way to think about the problem is to ask for each depth separately:

1. Find all nodes at depth `0`.
2. Find all nodes at depth `1`.
3. Find all nodes at depth `2`.
4. Continue until no deeper nodes exist.

Conceptually, this could be done with a helper:

```python
def collect_at_depth(node, depth, values):
    if node is None:
        return
    if depth == 0:
        values.append(node.val)
        return
    collect_at_depth(node.left, depth - 1, values)
    collect_at_depth(node.right, depth - 1, values)
```

Then repeatedly call it:

```python
answer = []
depth = 0

while there are nodes at this depth:
    level = []
    collect_at_depth(root, depth, level)
    if level is empty:
        break
    answer.append(level)
    depth += 1
```

This is correct as an idea, but it repeats work.

For a deep tree, collecting depth `10` walks through many of the same ancestors that were already walked for depths `0` through `9`. In the worst case, this can become much more expensive than necessary.

The better question is:

> Can we process each node once, while still knowing where one level ends and the next level begins?

---

### 3. Key Observation: A Level Produces the Next Level

Look at the tree level by level:

```text
level 0: [3]
level 1: [9, 20]
level 2: [15, 7]
```

If we already know all nodes in the current level, then the next level is exactly their children:

```text
children of [3]      -> [9, 20]
children of [9, 20]  -> [15, 7]
children of [15, 7]  -> []
```

So we do not need to rediscover deeper levels from the root every time.

We can maintain a frontier:

```text
frontier = nodes that are waiting to be processed next
```

At the start:

```text
frontier = [root]
```

After processing that frontier, we append its children to become the next frontier.

This is exactly what a queue is good at:

```text
remove old nodes from the front
add newly discovered children to the back
```

A queue gives first-in, first-out order, so nodes discovered earlier are processed earlier. Since children are only discovered after their parents, the queue naturally processes nodes from smaller depth to larger depth.

---

### 4. The Queue and Level Invariant

The most important invariant is:

```text
At the start of each outer loop iteration, the queue contains exactly the nodes of one tree level, in left-to-right order.
```

If that invariant is true, then the algorithm is straightforward:

1. The current queue length is the number of nodes on this level.
2. Remove exactly that many nodes.
3. Record their values into one list.
4. Append their children to the back of the queue.
5. The appended children become the next level.

The subtle detail is step 1:

```python
level_size = len(queue)
```

This value must be saved before processing the level.

Why?

Because while we process current-level nodes, we also enqueue their children. If we loop directly over the changing queue until it is empty, we will accidentally process children in the same level as their parents.

Freezing `level_size` separates the two responsibilities:

```text
first level_size dequeues  -> current level
new enqueues during loop   -> next level
```

That is the entire trick behind level order traversal.

---

### 5. Detailed Algorithm

Handle the empty tree first:

```text
if root is None, return []
```

Otherwise:

1. Create an empty answer list.
2. Create a queue containing only `root`.
3. While the queue is not empty:
   1. Save `level_size = len(queue)`.
   2. Create an empty `level` list.
   3. Repeat `level_size` times:
      1. Pop one node from the front of the queue.
      2. Append `node.val` to `level`.
      3. If `node.left` exists, append it to the back of the queue.
      4. If `node.right` exists, append it to the back of the queue.
   4. Append `level` to `answer`.
4. Return `answer`.

In Python, use `collections.deque` rather than a normal list for the queue.

A list can remove from the front with `pop(0)`, but that operation shifts all remaining elements and costs `O(n)`. A `deque` supports efficient front removal with `popleft()`.

---

### 6. Pseudocode

```python
from collections import deque


def level_order(root):
    if root is None:
        return []

    answer = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        answer.append(level)

    return answer
```

The code is short because the invariant does most of the work.

---

### 7. Example Walkthrough

Use Example 1:

```text
root = [3, 9, 20, None, None, 15, 7]
```

Tree form:

```text
        3
       / \
      9   20
         /  \
        15   7
```

Start:

```text
answer = []
queue  = [3]
```

#### Iteration 1

At the start, the queue contains exactly level `0`:

```text
queue = [3]
level_size = 1
level = []
```

Process one node:

```text
pop 3
level = [3]
enqueue 9
enqueue 20
queue = [9, 20]
```

Append the finished level:

```text
answer = [[3]]
```

Now the queue contains exactly the next level.

#### Iteration 2

At the start, the queue contains level `1`:

```text
queue = [9, 20]
level_size = 2
level = []
```

Process `9`:

```text
pop 9
level = [9]
9 has no children
queue = [20]
```

Process `20`:

```text
pop 20
level = [9, 20]
enqueue 15
enqueue 7
queue = [15, 7]
```

Append the finished level:

```text
answer = [[3], [9, 20]]
```

#### Iteration 3

At the start, the queue contains level `2`:

```text
queue = [15, 7]
level_size = 2
level = []
```

Process both nodes:

```text
pop 15
level = [15]
15 has no children

pop 7
level = [15, 7]
7 has no children

queue = []
```

Append the finished level:

```text
answer = [[3], [9, 20], [15, 7]]
```

The queue is now empty, so all reachable nodes have been processed.

Return:

```text
[[3], [9, 20], [15, 7]]
```

---

### 8. Why the Algorithm Is Correct

We prove correctness using the queue invariant.

#### Invariant

At the start of each outer `while` loop iteration:

```text
the queue contains exactly the nodes of the next unprocessed level, in left-to-right order
```

#### Initialization

Before the first iteration, the queue contains only `root`.

The root is exactly level `0`, and it is trivially in left-to-right order.

So the invariant is true before the loop begins.

#### Maintenance

Assume the invariant is true at the start of an iteration.

Let `level_size` be the queue length at that moment.

By the invariant, those `level_size` nodes are exactly the current level. The algorithm pops exactly those nodes and appends their values to `level`, so `level` contains exactly the current level's values in left-to-right order.

While processing those nodes, the algorithm appends each existing left child before each existing right child. Since parents are processed from left to right, their children are also appended in the correct left-to-right order for the next depth.

No other nodes are appended.

Therefore, after the `level_size` pops finish, the queue contains exactly the next level in left-to-right order.

So the invariant is preserved.

#### Termination

The loop ends when the queue is empty.

By the invariant, an empty queue means there is no next unprocessed level. Every node reachable from the root has already been processed exactly once, and each processed level has been appended to `answer`.

Therefore, `answer` contains all tree levels from top to bottom, with each level listed left to right. That is exactly the required output.

---

### 9. Complexity

Let `n` be the number of nodes in the tree.

#### Time Complexity

Each node is:

```text
enqueued once
dequeued once
visited once
```

All work done for a node is constant: read its value, check its two child pointers, and possibly enqueue children.

So the time complexity is:

```text
O(n)
```

#### Space Complexity

The result list stores all node values, which takes `O(n)` space as part of the output.

Ignoring the output, the queue stores at most one level plus some children of that level while transitioning to the next level. Its maximum size is proportional to the maximum width of the tree.

Let `w` be the maximum number of nodes on any level.

Auxiliary space is:

```text
O(w)
```

In the worst case, a complete binary tree can have about `n / 2` nodes on the last level, so `w` can be `O(n)`.

---

### 10. Common Pitfalls

#### Pitfall 1: Not Handling the Empty Tree

If `root` is `None`, the correct answer is:

```text
[]
```

Do not enqueue `None` and then try to read `None.val`.

#### Pitfall 2: Not Freezing the Level Size

This is the most common bug.

Incorrect idea:

```python
while queue:
    node = queue.popleft()
    # enqueue children
```

That produces a flat traversal, not grouped levels.

For this problem, every outer iteration must represent one level, so use:

```python
level_size = len(queue)
for _ in range(level_size):
    ...
```

#### Pitfall 3: Appending Children Before Recording the Current Level Size

The level boundary is defined by the queue contents at the start of the level. If children are added before measuring the level, the boundary is already corrupted.

Correct order:

```text
measure current level size
then process that many nodes
then leave children for the next iteration
```

#### Pitfall 4: Using `list.pop(0)` for the Queue

`pop(0)` works functionally, but it is inefficient because it shifts all remaining elements.

Prefer:

```python
from collections import deque
queue.popleft()
```

#### Pitfall 5: Mixing DFS Thinking With BFS Output

DFS can solve this problem by passing a `depth` parameter and appending into `answer[depth]`, but the first-principles BFS solution is simpler here because the desired output is already organized by levels.

If using BFS, let the queue boundary define the level boundary.

---

### 11. First-Principles Summary

The problem asks for nodes grouped by depth.

Depth increases one edge at a time, so once a level is known, the next level is exactly the children of that level's nodes.

A queue preserves that progression:

```text
current level leaves the front
next level enters the back
```

The only extra idea needed is to freeze the queue length before processing a level:

```text
level_size = len(queue)
```

That frozen size prevents newly enqueued children from being mixed into their parents' level.

So the essence of the solution is:

```text
repeat:
    process exactly the current frontier
    record its values as one level
    build the next frontier from its children
```

When the frontier becomes empty, there are no more levels, and the collected lists are the final level order traversal.

## Implementation
See `solutions/binary_tree_bfs/p102_binary_tree_level_order_traversal.py`.

## Tests
See `tests/binary_tree_bfs/test_p102_binary_tree_level_order_traversal.py`.

## Examples

### Example 1
- Input: `{'root': [3, 9, 20, None, None, 15, 7]}`
- Output: `[[3], [9, 20], [15, 7]]`

### Example 2
- Input: `{'root': [1]}`
- Output: `[[1]]`

### Example 3
- Input: `{'root': []}`
- Output: `[]`

## Follow-up Practice
- Trace the queue level by level.
- Implement with `deque` instead of a list pop from the front.
- Modify the level aggregation to produce sums, averages, or right-side values.
